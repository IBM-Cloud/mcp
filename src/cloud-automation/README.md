# IBMCloud MCP Server - Cloud Automation

This project provides a Model Context Protocol (MCP) server with tools for cloud automation services on IBM Cloud including Projects, Schematics and Deployable Architectures

## ⚙️ Configurations

The following example MCP configurations can be found in the `configs` folder to help get started with code engine use cases:

- [mcp.iac.json](https://github.com/IBM-Cloud/ibmcloud-mcp-server/blob/main/src/cloud-automation/configs/mcp.iac.json) - Tools for working with Cloud Automation, including Projects, Deployable Architectures, Private Catalogs and Schematics (Terraform/Ansible service).

Each of the above configurations also includes the following essential tools:

- `assist` - Detailed IBM Cloud Assistance "Assist me with IBMCLOUD_TOPIC..."
- `resource-groups` - Getting lists of the available resource groups
- `catalog_locations` - Getting available regions
- `target` - Change the account, resource group, region or cloud API endpoint being used as the current working context for tools.

## 🗣️ Prompts

The MCP Prompt feature is not currently implemented in the IBM Cloud MCP Servers, however, here are some prompts that you can 
use to get started using the tools:

### 🤖 Cloud Automation

- 🗣️ What projects are in this account?
- 🗣️ Create a new project, "`PROJECT_NAME`".
- 🗣️ List the environments in project, "`PROJECT_NAME`".
- 🗣️ Create new "dev", "test" and "prod" environments in `PROJECT_NAME`.
- 🗣️ List the configurations in project, "`PROJECT_NAME`".
- 🗣️ List the configuration version for the configuration, `CONFIG_ID`.
- 🗣️ List all the deployable architectures in all catalogs
- 🗣️ Show me detailed deployment information for the XYZ deployable architectures in the Community Registry catalog
- 🗣️ Select the `CODE_ENGINE_PROJECT_NAME` project to work with

- 🚧 TBD - WORK IN PROGRESS 🚧 
- 🗣️ What input values still need to be specified before I can deploy configuration `CONFIG_ID`?
- 🗣️ Validate the `CONFIG_ID` configuration.
- 🗣️ View the validation logs for `CONFIG_ID`
- 🗣️ Deploy `CONFIG_ID`
- 🗣️ Find errors in the deployment logs for project configurations.
- 🗣️ What can I try to resolve these deployment errors?

### Common

- 🗣️ Assist me with `IBM_CLOUD_TOPIC`
- 🗣️ What resource groups are in my account?
- 🗣️ Target the `RESOURCE_GROUP_NAME` resource group

> NOTE: The above common commands are from the [core](servers/core.md) set of tools and included in the iac configuration examples above.  There are many other core commands for working with access, accounts, users, services, resource searching, etc. that can be used simply by adding the tool names to the config.

## 🧾 Prerequisites

To build, run, and deploy this project, you will need the following installed:

* `make`
* Containerization tool: `podman` or `docker`
* `curl` (for testing)
* [`hadolint`](https://github.com/hadolint/hadolint?tab=readme-ov-file#install) (for linting Containerfiles)
* [`dockle`](https://github.com/goodwithtech/dockle) (for linting container images)
* [`trivy`](https://trivy.dev/v0.60/getting-started/installation/) (for scanning container images for vulnerabilities)
* IBM Cloud CLI with `container-registry` and `container-service` plugins

## 🗜️ Building the Container Image

The project uses a `Containerfile` to define the container image. You can build the image using either Podman or Docker via the Makefile targets.

An optional paramater of `IBMCLOUD_PLUGINS=one,two,three` can be added to the following build commands. All plugins will be installed if this is not provided.

* **Using Podman (Production image):**

    ```bash
    make podman
    ```

* **Using Podman (Development image):**

    ```bash
    make podman-dev
    ```

* **Using Docker (Production image):**

    ```bash
    make docker
    ```

* **Using Docker (Development image):**

    ```bash
    make docker-dev
    ```

## 🏃🏽‍♀️ Running the Container with an MCP Client

```json
{
    "mcpServers": {
        "ibmcloud-core": {
            "command": "/opt/podman/bin/podman",
            "args": [
                "run",
                "--platform=linux/arm64",
                "-p",
                "4444:4444",
                "-i",
                "--rm",
                "-e",
                "IBMCLOUD_API_KEY",
                "-e",
                "IBMCLOUD_MCP_TOOLS",
                "ibmcloud-mcpserver/ibmcloud-mcpserver"
            ],
            "env": {
                "IBMCLOUD_API_KEY": "<Your API key",
                "IBMCLOUD_MCP_TOOLS": "comma,separated,list,of,tools,to,enable"
            }
        }
    }
}
```

## 🏃🏽‍♀️ Running the Container Locally

You can run the built container image locally using the `make` targets for Podman or Docker.

Ensure you have a `.env` file in the project root. Copy `.env.example` to `.env` and set the values.

* **Running with Podman (HTTP on port 4444):**

    ```bash
    make podman-run
    ```

* **Running with Podman (HTTPS on port 4444):**

    ```bash
    make podman-run-ssl
    ```

* **Running with Docker (HTTP on port 4444):**

    ```bash
    make docker-run
    ```

* **Running with Docker (HTTPS on port 4444):**

    ```bash
    make docker-run-ssl
    ```

## ✋🏼 Stopping the Container

Stop and remove the running container instance:

* **Using Podman:**

    ```bash
    make podman-stop
    ```

* **Using Docker:**

    ```bash
    make docker-stop
    ```

## 🧪 Testing the Container

Use `curl` to quickly test the running container endpoint:

* **Using Podman:**

    ```bash
    make podman-test
    ```

* **Using Docker:**

    ```bash
    make docker-test
    ```

## 👮🏼 Vulnerability Scanning

The Makefile includes targets for scanning your `Containerfile` and built images for vulnerabilities and best practices.

* **Scan image for CVEs (HIGH/CRITICAL) using Trivy:**

    ```bash
    make trivy
    ```

    *Requires Podman socket enabled (`systemctl --user enable --now podman.socket`)*

* **Lint container image using Dockle:**

    ```bash
    make dockle
    ```

* **Lint Containerfile(s) using Hadolint:**

    ```bash
    make hadolint
    ```

## ☁️ Deployment to IBM Cloud Code Engine

This project includes Makefile targets to help you deploy the container image to IBM Cloud Code Engine.

1. **Configure Environment Variables:**

    Create a `.env.ce` file in the project root and define the necessary IBM Cloud Code Engine configuration variables. Copy `.env.ce.example` to `.env.ce` and set the values.

2. **Check Environment Variables:**

    ```bash
    make ibmcloud-check-env
    ```

3. **Install IBM Cloud CLI and Plugins (if needed):**

    ```bash
    make ibmcloud-cli-install
    ```

4. **Log in to IBM Cloud:**

    ```bash
    make ibmcloud-login
    ```

5. **Target Code Engine Project:**

    ```bash
    make ibmcloud-ce-login
    ```

6. **Build (if not already built) and Tag the Image:**

    ```bash
    make podman # or docker
    make ibmcloud-tag
    ```

7. **Push the Image to IBM Container Registry:**

    ```bash
    make ibmcloud-push
    ```

8. **Deploy/Update the Application in Code Engine:**

    ```bash
    make ibmcloud-deploy
    ```

9. **Stream Logs:**

    ```bash
    make ibmcloud-ce-logs
    ```

10. **Get Application Status:**

    ```bash
    make ibmcloud-ce-status
    ```

11. **Delete the Application:**

    ```bash
    make ibmcloud-ce-rm
    ```

## Getting Help

Run the default `make` target or `make help` to see a list of all available commands and their descriptions:

```bash
make help
```
