import { useState } from "react"
import { createFileRoute } from "@tanstack/react-router"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Loader2, Search, Users, Award } from "lucide-react"
import { useEndorsements } from "@/hooks/useEndorsements"
import useUserSearch from "@/hooks/useUserSearch"
import { type UserPublic } from "@/client"

interface EndorsementData {
  id: string
  endorser_id: string
  endorsed_id: string
  confidence: number
  created_at: string
  updated_at: string
  user_email: string
  user_full_name: string | null
}

export const Route = createFileRoute("/_layout/endorsement")({
  component: Endorsement,
  head: () => ({
    meta: [
      {
        title: "Endorsements - RepuLink",
      },
    ],
  }),
})

function EndorsementPage() {
  const { getMyEndorsements, getMyEndorsers, createEndorsement } = useEndorsements()

  const [searchTerm, setSearchTerm] = useState("")
  const [selectedUser, setSelectedUser] = useState<UserPublic | null>(null)
  const [confidenceValue, setConfidenceValue] = useState(0.5)

  const { data: searchResults } = useUserSearch(searchTerm)
  const myEndorsements = getMyEndorsements.data || []
  const myEndorsers = getMyEndorsers.data || []
  const isLoadingEndorsements = getMyEndorsements.isLoading || getMyEndorsers.isLoading

  const showSearchResults = searchTerm.trim().length > 0 && (searchResults?.data || []).length > 0

  const handleSelectUser = (user: UserPublic) => {
    setSelectedUser(user)
    setSearchTerm("")
    setConfidenceValue(0.5)
  }

  const handleEndorseUser = async () => {
    if (!selectedUser) return
    
    try {
      await createEndorsement.mutateAsync({
        endorsed_id: selectedUser.id,
        confidence: confidenceValue,
      })
      setSelectedUser(null)
      setConfidenceValue(0.5)
    } catch (error) {
      console.error("Endorsement error:", error)
    }
  }

  return (
    <div className="flex flex-col gap-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Endorsements</h1>
        <p className="text-muted-foreground mt-2">
          Build credibility by endorsing users and showcasing your trust in others
        </p>
      </div>

      {/* Search Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            Search and Endorse Users
          </CardTitle>
          <CardDescription>Find and endorse users with a confidence value</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Find User</label>
            <div className="flex gap-2">
              <Input
                placeholder="Search by name or email..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="flex-1"
              />
              <Button
                type="button"
                variant="outline"
                size="icon"
                title="Search"
              >
                <Search className="w-4 h-4" />
              </Button>
            </div>
            
            {/* Search Results Dropdown */}
            {showSearchResults && (
              <div className="border rounded-lg divide-y bg-background shadow-sm">
                {(searchResults?.data || []).map((u: any) => (
                  <button
                    key={u.id}
                    type="button"
                    onClick={() => handleSelectUser(u)}
                    className="w-full text-left p-3 hover:bg-accent transition-colors"
                  >
                    <div className="font-medium">{u.full_name || u.email}</div>
                    <div className="text-sm text-muted-foreground">{u.email}</div>
                  </button>
                ))}
              </div>
            )}

            {searchTerm.trim().length > 0 && (searchResults?.data || []).length === 0 && (
              <div className="text-center py-8 text-muted-foreground text-sm">
                <p>No users found. Try a different search term.</p>
              </div>
            )}
          </div>

          {/* Selected User Card */}
          {selectedUser && (
            <Card className="bg-muted/50">
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div>
                    <p className="font-medium">{selectedUser.full_name || selectedUser.email}</p>
                    <p className="text-sm text-muted-foreground">{selectedUser.email}</p>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Confidence Level</label>
                    <div className="flex items-center gap-4">
                      <Slider
                        value={[confidenceValue]}
                        onValueChange={(val: number[]) => setConfidenceValue(val[0])}
                        min={0}
                        max={1}
                        step={0.01}
                        className="flex-1"
                      />
                      <span className="text-sm font-medium min-w-12 text-right">
                        {confidenceValue.toFixed(2)}
                      </span>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button
                      onClick={handleEndorseUser}
                      disabled={createEndorsement.isPending}
                      className="flex-1"
                    >
                      {createEndorsement.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Endorsing...
                        </>
                      ) : (
                        <>
                          <Award className="mr-2 h-4 w-4" />
                          Endorse
                        </>
                      )}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => setSelectedUser(null)}
                      disabled={createEndorsement.isPending}
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </CardContent>
      </Card>

      {/* Endorsements Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Users I've Endorsed */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Award className="h-5 w-5" />
              Users I've Endorsed
            </CardTitle>
            <CardDescription>
              {myEndorsements.length} {myEndorsements.length === 1 ? "user" : "users"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoadingEndorsements ? (
              <div className="flex justify-center py-6">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : myEndorsements.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <p>You haven't endorsed anyone yet</p>
              </div>
            ) : (
              <div className="space-y-3">
                {myEndorsements.map((endorsement: EndorsementData) => (
                  <div
                    key={endorsement.id}
                    className="flex items-center justify-between p-3 rounded-lg border bg-card"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">
                        {endorsement.user_full_name || endorsement.user_email}
                      </p>
                      <p className="text-xs text-muted-foreground truncate">
                        {endorsement.user_email}
                      </p>
                    </div>
                    <div className="text-right ml-4 flex-shrink-0">
                      <p className="text-sm font-semibold">
                        {endorsement.confidence.toFixed(2)}
                      </p>
                      <p className="text-xs text-muted-foreground">confidence</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* My Endorsers */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              My Endorsers
            </CardTitle>
            <CardDescription>
              {myEndorsers.length} {myEndorsers.length === 1 ? "endorser" : "endorsers"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoadingEndorsements ? (
              <div className="flex justify-center py-6">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : myEndorsers.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <p>No one has endorsed you yet</p>
              </div>
            ) : (
              <div className="space-y-3">
                {myEndorsers.map((endorsement: EndorsementData) => (
                  <div
                    key={endorsement.id}
                    className="flex items-center justify-between p-3 rounded-lg border bg-card"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">
                        {endorsement.user_full_name || endorsement.user_email}
                      </p>
                      <p className="text-xs text-muted-foreground truncate">
                        {endorsement.user_email}
                      </p>
                    </div>
                    <div className="text-right ml-4 flex-shrink-0">
                      <p className="text-sm font-semibold">
                        {endorsement.confidence.toFixed(2)}
                      </p>
                      <p className="text-xs text-muted-foreground">confidence</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function Endorsement() {
  return <EndorsementPage />
}
