import {
  MutationCache,
  QueryCache,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query"
import { createRouter, RouterProvider } from "@tanstack/react-router"
import { StrictMode } from "react"
import ReactDOM from "react-dom/client"
import { ApiError, OpenAPI } from "./client"
import { ThemeProvider } from "./components/theme-provider"
import { Toaster } from "./components/ui/sonner"
import "./index.css"
import { routeTree } from "./routeTree.gen"
import { TOOLBOX_ORIGIN } from "./constants"

const isAllowedOrigin = (origin: string) => origin === TOOLBOX_ORIGIN

let pendingTokenRefresh: Promise<string> | null = null

const requestFreshToken = (): Promise<string> => {
  if (pendingTokenRefresh) return pendingTokenRefresh

  pendingTokenRefresh = new Promise<string>((resolve) => {
    const handler = (event: Event) => {
      pendingTokenRefresh = null
      resolve((event as CustomEvent<{ token: string }>).detail.token)
    }
    window.addEventListener("sso-token-refreshed", handler, { once: true })
    window.parent.postMessage({ type: "IFRAME_REQUEST_TOKEN" }, "*")
    setTimeout(() => {
      window.removeEventListener("sso-token-refreshed", handler)
      pendingTokenRefresh = null
      resolve(localStorage.getItem("access_token") || "")
    }, 5000)
  })

  return pendingTokenRefresh
}

OpenAPI.BASE = import.meta.env.VITE_API_URL
OpenAPI.TOKEN = async () => {
  const token = localStorage.getItem("access_token")
  if (!token) return ""

  if (window.parent !== window) {
    try {
      const base64url = token.split(".")[1]
      const base64 = base64url.replace(/-/g, "+").replace(/_/g, "/")
      const padded = base64.padEnd(base64.length + (4 - (base64.length % 4)) % 4, "=")
      const payload = JSON.parse(atob(padded))
      const expiresIn = payload.exp - Math.floor(Date.now() / 1000)
      if (expiresIn < 30) return requestFreshToken()
    } catch {
      // not a JWT, use as-is
    }
  }

  return token
}

const handleApiError = (error: Error) => {
  if (error instanceof ApiError && [401, 403].includes(error.status)) {
    if (window.parent !== window) {
      window.parent.postMessage({ type: "IFRAME_REQUEST_TOKEN" }, "*")
    } else {
      localStorage.removeItem("access_token")
      window.location.href = "/login"
    }
  }
}

const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: handleApiError,
  }),
  mutationCache: new MutationCache({
    onError: handleApiError,
  }),
})

window.addEventListener("message", (event) => {
  if (!isAllowedOrigin(event.origin)) return
  if (event.data?.type !== "SSO_TOKEN") return
  localStorage.setItem("access_token", event.data.token)
  window.dispatchEvent(
    new CustomEvent("sso-token-refreshed", { detail: { token: event.data.token } }),
  )
  queryClient.invalidateQueries()
})

const router = createRouter({ routeTree })
declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
        <Toaster richColors closeButton />
      </QueryClientProvider>
    </ThemeProvider>
  </StrictMode>,
)