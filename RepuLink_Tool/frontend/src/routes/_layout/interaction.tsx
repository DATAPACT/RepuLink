import { useState } from "react"
import { createFileRoute } from "@tanstack/react-router"
import { Check, X, Send, Star, User, Search } from "lucide-react"
import useAuth from "@/hooks/useAuth"
import useCustomToast from "@/hooks/useCustomToast"
import { useCreateInteraction, useUserInteractions, useRespondInteraction } from "@/hooks/useInteractions"
import useUserSearch from "@/hooks/useUserSearch"
import useAddRating, { useRatingsForInteraction } from "@/hooks/useRatings"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"

export const Route = createFileRoute("/_layout/interaction")({
  component: InteractionPage,
  head: () => ({
    meta: [
      {
        title: "Interactions - RepuLink",
      },
    ],
  }),
})

function InteractionPage(): React.JSX.Element {
  const { user } = useAuth()
  const [targetId, setTargetId] = useState("")
  const [message, setMessage] = useState("")
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedUser, setSelectedUser] = useState<any>(null)
  const [expandedId, setExpandedId] = useState<string | null>(null)

  console.log("InteractionPage: user =", user)

  const createMutation = useCreateInteraction()
  const { data: interactions, isLoading, error } = useUserInteractions(user?.id)
  const { data: searchResults } = useUserSearch(searchTerm)

  console.log("InteractionPage: interactions =", interactions, "isLoading =", isLoading, "error =", error)

  const handleSelectUser = (selectedUser: any) => {
    setTargetId(selectedUser.id)
    setSelectedUser(selectedUser)
    setSearchTerm("")
  }

  const showSearchResults = searchTerm.trim().length > 0 && (searchResults?.data || []).length > 0

  const handleSendRequest = (e: React.FormEvent) => {
    e.preventDefault()
    if (!targetId) return
    createMutation.mutate({ target_id: targetId, message })
    setMessage("")
    setTargetId("")
    setSelectedUser(null)
  }

  return (
    <div className="flex flex-col gap-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Interactions</h1>
        <p className="text-muted-foreground mt-2">
          Build trust by sending and responding to interaction requests
        </p>
      </div>

      {/* Send Request Card */}
      <Card>
        <CardHeader>
          <CardTitle>Send Interaction Request</CardTitle>
          <CardDescription>Connect with other users and build relationships</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSendRequest} className="space-y-4">
            {/* Search Users */}
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
                      className="w-full text-left px-4 py-3 hover:bg-muted transition-colors flex items-center gap-3"
                    >
                      <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                        <User className="w-4 h-4" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium truncate">{u.full_name || "Unknown"}</div>
                        <div className="text-sm text-muted-foreground truncate">{u.email}</div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
              
              {/* Empty State */}
              {searchTerm.trim().length > 0 && (!searchResults?.data || searchResults.data.length === 0) && (
                <div className="border rounded-lg bg-muted/50 px-4 py-6 text-center">
                  <p className="text-sm text-muted-foreground">
                    No users found matching "{searchTerm}"
                  </p>
                </div>
              )}
            </div>

            {/* Selected User Display */}
            {selectedUser && (
              <Card className="bg-muted/50">
                <CardContent className="pt-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <User className="w-5 h-5" />
                      </div>
                      <div>
                        <div className="font-medium">{selectedUser.full_name || "Unknown"}</div>
                        <div className="text-sm text-muted-foreground">{selectedUser.email}</div>
                      </div>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setSelectedUser(null)
                        setTargetId("")
                      }}
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Message */}
            <div>
              <label className="text-sm font-medium">Message</label>
              <Textarea
                placeholder="Add a message to your request... (optional)"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                className="mt-1"
                rows={4}
              />
            </div>

            {/* Submit Button */}
            <Button type="submit" disabled={!targetId || createMutation.isPending} className="w-full">
              <Send className="w-4 h-4 mr-2" />
              {createMutation.isPending ? "Sending..." : "Send Request"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Interactions List */}
      <div>
        <h2 className="text-xl font-bold tracking-tight mb-4">Your Interactions</h2>
        {isLoading && (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12 text-center">
              <div className="text-sm text-muted-foreground">Loading interactions...</div>
            </CardContent>
          </Card>
        )}
        {error && (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12 text-center">
              <div className="text-sm text-destructive">Error loading interactions: {error.message}</div>
            </CardContent>
          </Card>
        )}
        {!isLoading && !error && (!interactions || interactions.length === 0) && (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12 text-center">
              <div className="rounded-full bg-muted p-4 mb-4">
                <User className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-semibold">No interactions yet</h3>
              <p className="text-muted-foreground mt-1">
                Send your first interaction request to get started
              </p>
            </CardContent>
          </Card>
        )}
        {!isLoading && !error && interactions && interactions.length > 0 && (
          <div className="space-y-4">
            {/* Interactions Table */}
            <div className="border rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-muted border-b">
                  <tr>
                    <th className="px-4 py-3 text-left font-medium">Type</th>
                    <th className="px-4 py-3 text-left font-medium">User</th>
                    <th className="px-4 py-3 text-left font-medium">Status</th>
                    <th className="px-4 py-3 text-left font-medium">Date</th>
                    <th className="px-4 py-3 text-left font-medium">Ratings</th>
                    <th className="px-4 py-3 text-left font-medium">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {interactions.map((interaction: any) => (
                    <InteractionTableRow
                      key={interaction.id}
                      interaction={interaction}
                      user={user}
                      isExpanded={expandedId === interaction.id}
                      onToggle={() => setExpandedId(expandedId === interaction.id ? null : interaction.id)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function InteractionTableRow({
  interaction,
  user,
  isExpanded,
  onToggle,
}: {
  interaction: any
  user: any
  isExpanded: boolean
  onToggle: () => void
}): React.JSX.Element {
  const getStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      pending: "secondary",
      accepted: "default",
      denied: "destructive",
    }
    return <Badge variant={variants[status] || "outline"}>{status}</Badge>
  }

  return (
    <>
      <tr className="border-b hover:bg-muted/50 cursor-pointer" onClick={onToggle}>
        <td className="px-4 py-3">{interaction.initiator_id === user?.id ? "Sent" : "Received"}</td>
        <td className="px-4 py-3 text-primary font-medium">{(interaction.initiator_id === user?.id ? interaction.target_id : interaction.initiator_id).slice(0, 8)}...</td>
        <td className="px-4 py-3">{getStatusBadge(interaction.status)}</td>
        <td className="px-4 py-3 text-muted-foreground text-xs">
          {new Date(interaction.created_at).toLocaleDateString()}
        </td>
        <td className="px-4 py-3 text-xs text-muted-foreground">
          {/* Ratings count will be shown when expanded */}
          <span>-</span>
        </td>
        <td className="px-4 py-3 text-right">
          <Button variant="ghost" size="sm" onClick={(e) => {
            e.stopPropagation()
            onToggle()
          }}>
            {isExpanded ? "Hide" : "Show"}
          </Button>
        </td>
      </tr>
      {isExpanded && (
        <tr className="border-b bg-muted/30">
          <td colSpan={6} className="px-4 py-4">
            <InteractionDetails interaction={interaction} user={user} />
          </td>
        </tr>
      )}
    </>
  )
}

function InteractionDetails({ interaction, user }: { interaction: any; user: any }): React.JSX.Element {
  const respondMutation = useRespondInteraction()

  return (
    <div className="space-y-4">
      {/* Message */}
      {interaction.message && (
        <div>
          <h4 className="font-medium mb-2">Message</h4>
          <div className="bg-background p-3 rounded-lg border">
            <p className="text-sm">{interaction.message}</p>
          </div>
        </div>
      )}

      {/* Action Buttons - Pending */}
      {interaction.target_id === user?.id && interaction.status === "pending" && (
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="default"
            onClick={() => respondMutation.mutate({ id: interaction.id, accept: true })}
            disabled={respondMutation.isPending}
          >
            <Check className="w-4 h-4 mr-1" />
            Accept
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => respondMutation.mutate({ id: interaction.id, accept: false })}
            disabled={respondMutation.isPending}
          >
            <X className="w-4 h-4 mr-1" />
            Deny
          </Button>
        </div>
      )}

      {/* Rating Section - Accepted */}
      {interaction.status === "accepted" && (
        <>
          <RatingFormWithCheck interaction={interaction} />
          
          {/* Ratings Display - Show submitted ratings */}
          <InteractionRatingsDisplay interactionId={interaction.id} />
        </>
      )}
    </div>
  )
}

function RatingFormWithCheck({ interaction }: { interaction: any }): React.JSX.Element | null {
  const { user } = useAuth()
  const { data: ratings } = useRatingsForInteraction(interaction.id)
  
  const userHasRated = ratings?.some((r: any) => r.rater_id === user?.id) || false
  
  return <RatingForm interaction={interaction} userHasRated={userHasRated} />
}

function InteractionRatingsDisplay({ interactionId }: { interactionId: string }): React.JSX.Element {
  const { data: ratings, isLoading } = useRatingsForInteraction(interactionId)

  if (isLoading) {
    return <div className="text-sm text-muted-foreground">Loading ratings...</div>
  }

  if (!ratings || ratings.length === 0) {
    return <div />
  }

  return (
    <div className="border-t pt-4">
      <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
        <Star className="w-4 h-4" />
        Ratings ({ratings.length})
      </h4>
      <div className="space-y-3">
        {ratings.map((rating: any) => (
          <div key={rating.id} className="bg-muted/50 p-3 rounded-md space-y-1">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {rating.rating < 0 ? (
                  <>
                    {[...Array(Math.abs(rating.rating))].map((_, i) => (
                      <Star
                        key={`negative-${i}`}
                        className="w-3 h-3 fill-red-500 text-red-500"
                      />
                    ))}
                    {[...Array(5 - Math.abs(rating.rating))].map((_, i) => (
                      <Star
                        key={`empty-${i}`}
                        className="w-3 h-3 text-muted-foreground"
                      />
                    ))}
                  </>
                ) : (
                  <>
                    {[...Array(rating.rating)].map((_, i) => (
                      <Star
                        key={`positive-${i}`}
                        className="w-3 h-3 fill-yellow-400 text-yellow-400"
                      />
                    ))}
                    {[...Array(5 - rating.rating)].map((_, i) => (
                      <Star
                        key={`empty-${i}`}
                        className="w-3 h-3 text-muted-foreground"
                      />
                    ))}
                  </>
                )}
                <span className="text-sm font-medium">{rating.rating > 0 ? `+${rating.rating}` : rating.rating}/5</span>
              </div>
            </div>
            <div className="text-xs text-muted-foreground">
              <span className="font-medium">{rating.rater_full_name || rating.rater_email}</span>
              {" • "}
              {new Date(rating.created_at).toLocaleDateString()} {new Date(rating.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
            {rating.comment && <p className="text-sm text-foreground">{rating.comment}</p>}
          </div>
        ))}
      </div>
    </div>
  )
}

function RatingForm({ interaction, userHasRated }: { interaction: any; userHasRated: boolean }): React.JSX.Element | null {
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const [ratingState, setRatingState] = useState<{ rating: number; comment: string }>({ rating: 0, comment: "" })
  const addRatingMutation = useAddRating()

  if (userHasRated) {
    return null
  }

  const handleSubmit = () => {
    addRatingMutation.mutate(
      {
        interactionId: interaction.id,
        rating: ratingState.rating,
        comment: ratingState.comment,
      },
      {
        onSuccess: () => {
          showSuccessToast("Rating submitted successfully!")
          setRatingState({ rating: 0, comment: "" })
        },
        onError: (error: any) => {
          const errorMsg = error?.message || error?.detail || String(error) || "Failed to submit rating"
          showErrorToast(errorMsg)
        },
      }
    )
  }

  return (
    <div className="border-t pt-4">
      <div className="space-y-3">
        <h4 className="text-sm font-medium flex items-center gap-2">
          <Star className="w-4 h-4" />
          Rate this interaction
        </h4>
        <div className="space-y-2">
          <div>
            <label className="text-sm">Rating (-5 to 5)</label>
            <select
              value={ratingState.rating}
              onChange={(e) => setRatingState({ ...ratingState, rating: parseInt(e.target.value) })}
              className="w-full mt-1 px-3 py-2 border rounded-md text-sm"
            >
              {[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].map((n) => (
                <option key={n} value={n}>
                  {n > 0 ? `+${n}` : n}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-sm">Comment (optional)</label>
            <Textarea
              placeholder="Share your experience..."
              value={ratingState.comment}
              onChange={(e) => setRatingState({ ...ratingState, comment: e.target.value })}
              className="mt-1"
              rows={2}
            />
          </div>
          <Button
            size="sm"
            onClick={handleSubmit}
            disabled={addRatingMutation.isPending}
            className="w-full"
          >
            <Star className="w-4 h-4 mr-1" />
            {addRatingMutation.isPending ? "Submitting..." : "Submit Rating"}
          </Button>
        </div>
      </div>
    </div>
  )
}
