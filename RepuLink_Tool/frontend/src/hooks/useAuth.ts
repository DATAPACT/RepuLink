import { useEffect } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useNavigate } from "@tanstack/react-router"

import {
  type Body_login_login_access_token as AccessToken,
  LoginService,
  type UserPublic,
  type UserRegister,
  UsersService,
} from "@/client"
import { handleError } from "@/utils"
import useCustomToast from "./useCustomToast"

const isLoggedIn = () => {
  return localStorage.getItem("access_token") !== null
}

const getUserFromKeycloakToken = (): UserPublic | null => {
  const token = localStorage.getItem("access_token")
  if (!token) return null
  try {
    const base64url = token.split(".")[1]
    const base64 = base64url.replace(/-/g, "+").replace(/_/g, "/")
    const padded = base64.padEnd(base64.length + (4 - (base64.length % 4)) % 4, "=")
    const payload = JSON.parse(atob(padded))
    if (!payload.iss) return null  // internal JWT has no iss
    return {
      id: payload.sub,
      email: payload.email,
      full_name: payload.name ?? null,
      is_active: true,
      is_superuser: false,
    }
  } catch {
    return null
  }
}

const useAuth = () => {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { showErrorToast } = useCustomToast()

  useEffect(() => {
    const handler = (event: MessageEvent) => {
      const allowedOrigins = [".dp.assistcloud.net", "http://localhost:3000"]
      if (!allowedOrigins.some(o => event.origin === o || event.origin.endsWith(o))) return
      if (event.data?.type !== "SSO_TOKEN") return
      if (isLoggedIn()) return
      localStorage.setItem("access_token", event.data.token)
      navigate({ to: "/" })
    }
    window.addEventListener("message", handler)
    return () => window.removeEventListener("message", handler)
  }, [navigate])

  const ssoUser = getUserFromKeycloakToken()

  const { data: fetchedUser } = useQuery<UserPublic | null, Error>({
    queryKey: ["currentUser"],
    queryFn: UsersService.readUserMe,
    enabled: isLoggedIn() && !ssoUser,
  })

  const user = ssoUser ?? fetchedUser

  const signUpMutation = useMutation({
    mutationFn: (data: UserRegister) =>
      UsersService.registerUser({ requestBody: data }),
    onSuccess: () => {
      navigate({ to: "/login" })
    },
    onError: handleError.bind(showErrorToast),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] })
    },
  })

  const login = async (data: AccessToken) => {
    const response = await LoginService.loginAccessToken({
      formData: data,
    })
    localStorage.setItem("access_token", response.access_token)
  }

  const loginMutation = useMutation({
    mutationFn: login,
    onSuccess: () => {
      navigate({ to: "/" })
    },
    onError: handleError.bind(showErrorToast),
  })

  const logout = () => {
    localStorage.removeItem("access_token")
    navigate({ to: "/login" })
  }

  return {
    signUpMutation,
    loginMutation,
    logout,
    user,
  }
}

export { isLoggedIn }
export default useAuth
