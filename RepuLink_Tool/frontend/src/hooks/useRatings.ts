import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

const apiBase = import.meta.env.VITE_API_URL ?? ""

function authHeader() {
  const token = localStorage.getItem("access_token")
  const headers: Record<string, string> = {}
  if (token) headers.Authorization = `Bearer ${token}`
  return headers
}

export function useAddRating() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async ({ interactionId, rating, comment }: { interactionId: string; rating: number; comment?: string }) => {
      const res = await fetch(`${apiBase}/api/v1/interactions/${interactionId}/rating`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeader(),
        },
        body: JSON.stringify({ rating, comment }),
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

export function useRatingsForInteraction(interactionId?: string) {
  return useQuery({
    queryKey: ["ratings", interactionId],
    queryFn: async () => {
      if (!interactionId) return []
      const res = await fetch(`${apiBase}/api/v1/interactions/${interactionId}/ratings`, {
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
    enabled: Boolean(interactionId),
  })
}

export default useAddRating
