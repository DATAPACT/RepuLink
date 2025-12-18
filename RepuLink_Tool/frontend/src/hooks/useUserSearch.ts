import { useQuery } from "@tanstack/react-query"

const apiBase = import.meta.env.VITE_API_URL ?? ""

function authHeader() {
  const token = localStorage.getItem("access_token")
  const headers: Record<string, string> = {}
  if (token) headers.Authorization = `Bearer ${token}`
  return headers
}

export function useUserSearch(query: string | undefined) {
  return useQuery({
    queryKey: ["userSearch", query],
    queryFn: async () => {
      if (!query || query.trim() === "") return { data: [], count: 0 }
      const res = await fetch(`${apiBase}/api/v1/users/search?query=${encodeURIComponent(query)}`, {
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
    enabled: Boolean(query),
    staleTime: 1000 * 60 * 1,
  })
}

export default useUserSearch
