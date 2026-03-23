import { createFileRoute, Outlet, redirect } from "@tanstack/react-router"

import { TOOLBOX_ORIGIN } from "@/constants"
import { Footer } from "@/components/Common/Footer"
import AppSidebar from "@/components/Sidebar/AppSidebar"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { isLoggedIn } from "@/hooks/useAuth"

const waitForSsoToken = (): Promise<string | null> =>
  new Promise((resolve) => {
    const timeout = setTimeout(() => resolve(null), 3000)
    const handler = (event: MessageEvent) => {
      if (event.origin !== TOOLBOX_ORIGIN) return
      if (event.data?.type !== "SSO_TOKEN") return
      clearTimeout(timeout)
      resolve(event.data.token)
    }
    window.addEventListener("message", handler, { once: true })
    window.parent.postMessage({ type: "IFRAME_READY" }, "*")
  })

export const Route = createFileRoute("/_layout")({
  component: Layout,
  beforeLoad: async () => {
    if (isLoggedIn()) return

    if (window.parent !== window) {
      const token = await waitForSsoToken()
      if (token) {
        localStorage.setItem("access_token", token)
        return
      }
    }

    throw redirect({ to: "/login" })
  },
})

function Layout() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="sticky top-0 z-10 flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1 text-muted-foreground" />
        </header>
        <main className="flex-1 p-6 md:p-8">
          <div className="mx-auto max-w-7xl">
            <Outlet />
          </div>
        </main>
        <Footer />
      </SidebarInset>
    </SidebarProvider>
  )
}

export default Layout
