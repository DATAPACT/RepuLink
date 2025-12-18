import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"

const apiBase = import.meta.env.VITE_API_URL ?? ""

function authHeader() {
  const token = localStorage.getItem("access_token")
  const headers: Record<string, string> = {}
  if (token) headers.Authorization = `Bearer ${token}`
  return headers
}

export function useEndorsements() {
  const queryClient = useQueryClient()

  const getMyEndorsements = useQuery({
    queryKey: ["endorsements", "endorsed-by-me"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/api/v1/endorsements/endorsed-by-me`, {
        headers: authHeader(),
      })
      if (!res.ok) {
        throw new Error("Failed to fetch endorsements")
      }
      return res.json()
    },
    staleTime: 0,
    retry: 1,
  })

  const getMyEndorsers = useQuery({
    queryKey: ["endorsements", "endorsing-me"],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/api/v1/endorsements/endorsing-me`, {
        headers: authHeader(),
      })
      if (!res.ok) {
        throw new Error("Failed to fetch endorsers")
      }
      return res.json()
    },
    staleTime: 0,
    retry: 1,
  })

  const createEndorsement = useMutation({
    mutationFn: async (data: { endorsed_id: string; confidence: number }) => {
      const res = await fetch(`${apiBase}/api/v1/endorsements/`, {
        method: "POST",
        headers: {
          ...authHeader(),
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || "Failed to create endorsement")
      }
      return res.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["endorsements", "endorsed-by-me"] })
      queryClient.invalidateQueries({ queryKey: ["endorsements", "endorsing-me"] })
    },
  })

  return {
    getMyEndorsements,
    getMyEndorsers,
    createEndorsement,
  }
}


