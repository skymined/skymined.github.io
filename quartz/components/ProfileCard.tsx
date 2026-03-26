import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { isAbsoluteURL, joinSegments } from "../util/path"
import style from "./styles/profileCard.scss"

interface SocialLink {
  label: string
  href: string
  iconSrc?: string
  iconAlt?: string
}

interface Options {
  imageSrc?: string
  imageAlt?: string
  bio?: string
  links?: SocialLink[]
}

const defaultLinks: SocialLink[] = [
  { label: "in", href: "https://www.linkedin.com/in/your-id" },
  { label: "gh", href: "https://github.com/your-id" },
  { label: "mail", href: "mailto:you@example.com" },
]

export default ((userOpts?: Options) => {
  const withBasePath = (src: string | undefined, baseUrl?: string) => {
    if (!src || isAbsoluteURL(src) || !src.startsWith("/")) {
      return src
    }

    if (!baseUrl) {
      return src
    }

    const basePath = new URL(`https://${baseUrl}`).pathname
    return joinSegments(basePath, src)
  }

  const ProfileCard: QuartzComponent = ({ displayClass, cfg }: QuartzComponentProps) => {
    const imageSrc = withBasePath(userOpts?.imageSrc, cfg.baseUrl)
    const imageAlt = userOpts?.imageAlt ?? "Profile image"
    const bio =
      userOpts?.bio ??
      "Write a short bio here. Add your profile image and social links in quartz.layout.ts."
    const links = userOpts?.links ?? defaultLinks

    return (
      <section class={classNames(displayClass, "profile-card")}>
        {imageSrc ? (
          <img class="profile-image" src={imageSrc} alt={imageAlt} loading="lazy" />
        ) : (
          <div class="profile-image profile-image--placeholder">{imageAlt}</div>
        )}
        <p class="profile-bio">{bio}</p>
        <ul class="profile-links">
          {links.map((link) => {
            const external = /^https?:\/\//.test(link.href)
            return (
              <li>
                <a
                  href={link.href}
                  target={external ? "_blank" : undefined}
                  rel={external ? "noopener noreferrer" : undefined}
                  aria-label={link.label}
                >
                  {link.iconSrc ? (
                    <img
                      class="profile-link-icon"
                      src={withBasePath(link.iconSrc, cfg.baseUrl)}
                      alt={link.iconAlt ?? link.label}
                      loading="lazy"
                    />
                  ) : (
                    link.label
                  )}
                </a>
              </li>
            )
          })}
        </ul>
      </section>
    )
  }

  ProfileCard.css = style
  return ProfileCard
}) satisfies QuartzComponentConstructor
