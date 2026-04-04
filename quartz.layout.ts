import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

const profileCard = Component.ProfileCard({
  // Put files under quartz/static/, then use /static/... URL here
  imageSrc: "/static/pics/title1.jpg",
  imageAlt: "Profile image",
  bio: "I write archive-first notes and development logs.",
  links: [
    {
      label: "LinkedIn",
      href: "https://www.linkedin.com/in/sky-kim-545b5a303/",
      // iconSrc: "/static/social/linkedin.png",
      // iconAlt: "LinkedIn",
    },
    {
      label: "GitHub",
      href: "https://github.com/skymined",
      // iconSrc: "/static/social/github.png",
      // iconAlt: "GitHub",
    },
    {
      label: "Mail",
      href: "mailto:adsky0309@korea.ac.kr",
      // iconSrc: "/static/social/mail.png",
      // iconAlt: "Mail",
    },
  ],
})

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/jackyzha0/quartz",
      "Discord Community": "https://discord.gg/cRFFHYye7t",
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.MobileOnly(profileCard),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [
    Component.PageTitle(),
    Component.DesktopOnly(profileCard),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
        { Component: Component.ReaderMode() },
      ],
    }),
    Component.Explorer({
      folderDefaultState: "collapsed",
      useSavedState: true,
      filterFn: (node) => node.isFolder && node.slugSegment !== "tags",
    }),
  ],
  right: [
    Component.Graph(),
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Backlinks(),
  ],
}

// components for pages that display lists of pages (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs(),
    Component.MobileOnly(profileCard),
    Component.ArticleTitle(),
    Component.ContentMeta(),
  ],
  left: [
    Component.PageTitle(),
    Component.DesktopOnly(profileCard),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.Explorer({
      folderDefaultState: "collapsed",
      useSavedState: true,
      filterFn: (node) => node.isFolder && node.slugSegment !== "tags",
    }),
  ],
  right: [],
}
