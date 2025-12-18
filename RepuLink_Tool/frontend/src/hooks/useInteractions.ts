import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

const apiBase = import.meta.env.VITE_API_URL ?? ""

function authHeader() {
  const token = localStorage.getItem("access_token")
  const headers: Record<string, string> = {}
  if (token) headers.Authorization = `Bearer ${token}`
  return headers
}

export function useCreateInteraction() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (data: { target_id: string; message?: string }) => {
      const res = await fetch(`${apiBase}/api/v1/interactions/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeader(),
        },
        body: JSON.stringify(data),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText)
      }
      return res.json()
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["interactions"] }),
  })
}

export function useRespondInteraction() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async ({ id, accept }: { id: string; accept: boolean }) => {
      const res = await fetch(`${apiBase}/api/v1/interactions/${id}/respond?accept=${accept}`, {
        method: "POST",
        headers: {
          ...authHeader(),
        },
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText)
      }
      return res.json()
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["interactions"] }),
  })
}

export function useUserInteractions(userId?: string) {
  return useQuery({
    queryKey: ["interactions", userId],
    queryFn: async () => {
      if (!userId) {
        console.log("useUserInteractions: userId is empty, returning empty array")
        return []
      }
      console.log(`useUserInteractions: Fetching interactions for user ${userId}`)
      const res = await fetch(`${apiBase}/api/v1/interactions/users/${userId}`, {
        headers: {
          ...authHeader(),
        },
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText)
      }
      const data = await res.json()
      console.log(`useUserInteractions: Got ${data?.length || 0} interactions`, data)
      return data
    },
    enabled: Boolean(userId),
    staleTime: 0,
    refetchOnMount: true,
  })
}

export default useCreateInteraction
