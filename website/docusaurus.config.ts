import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Project Eureka',
  tagline: 'AI Frontiers Evaluation and Understanding',
  favicon: 'img/eureka_logo.png',

  // Set the production url of your site here
  url: 'https://microsoft.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/eureka-ml-insights/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'Microsoft', // Usually your GitHub org/user name.
  projectName: 'eureka-ml-insights', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/microsoft/eureka-ml-insights',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://aka.ms/eureka-ml-insights-blog',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    // image: 'img/background.png',
    navbar: {
      title: 'Eureka Model Benchmarks',
      logo: {
        alt: 'Project Eureka',
        src: 'img/eureka_logo.png',
      },
      items: [
        {
          href: 'https://aka.ms/eureka-ml-insights-blog',
          label: 'Blog', 
          position: 'right'
        },
        {
          href: 'https://github.com/microsoft/eureka-ml-insights',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          label: 'Blog',
          href: 'https://aka.ms/eureka-ml-insights-blog',
        },
        {
          label: 'GitHub',
          href: 'https://github.com/microsoft/eureka-ml-insights',
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Microsoft Research | 
        <a target="_blank" style="color:#10adff" href="https://go.microsoft.com/fwlink/?LinkId=521839">Privacy and Cookies</a> |  
        <a target="_blank" style="color:#10adff" href="https://go.microsoft.com/fwlink/?linkid=2259814">Consumer Health Privacy</a> |  
        <a target=_blank style="color:#10adff" href="https://go.microsoft.com/fwlink/?LinkID=206977">Terms of Use</a> | 
        <a target="_blank" style="color:#10adff" href="mailto:eureka-ml-insights@microsoft.com">Contact Us</a> | 
        <a target="_blank" style="color:#10adff" href="https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks">Trademarks</a>`,
    },
    colorMode: {
      disableSwitch: true,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
