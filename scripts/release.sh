#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FWBG_DIR="$(dirname "$SCRIPT_DIR")"
DASHBOARD_DIR="${FWBG_DIR}/../fwbg-dashboard"

# Version type from argument
VERSION_TYPE=$1

if [[ ! "$VERSION_TYPE" =~ ^(patch|minor|major)$ ]]; then
    echo -e "${RED}Usage: ./scripts/release.sh <patch|minor|major>${NC}"
    exit 1
fi

# Function to get current version from git tags
get_current_version() {
    local dir=$1
    cd "$dir"
    local latest_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
    echo "${latest_tag#v}"
}

# Function to calculate new version
calculate_new_version() {
    local current=$1
    local type=$2

    IFS='.' read -r major minor patch <<< "$current"

    case $type in
        major)
            echo "$((major + 1)).0.0"
            ;;
        minor)
            echo "$major.$((minor + 1)).0"
            ;;
        patch)
            echo "$major.$minor.$((patch + 1))"
            ;;
    esac
}

# Function to release a project
release_project() {
    local name=$1
    local dir=$2
    local version=$3

    echo -e "\n${BLUE}📦 Releasing $name v$version${NC}"

    cd "$dir"

    # Check for uncommitted changes
    if [[ -n $(git status --porcelain) ]]; then
        echo -e "${YELLOW}⚠️  $name has uncommitted changes. Committing...${NC}"
        git add -A
        git commit -m "Prepare release v$version" || true
    fi

    # Create and push tag
    if git rev-parse "v$version" >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Tag v$version already exists in $name, skipping...${NC}"
    else
        git tag "v$version"
        echo -e "${GREEN}✅ Created tag v$version${NC}"

        git push origin main
        git push origin "v$version"
        echo -e "${GREEN}✅ Pushed changes and tag${NC}"
    fi
}

# Get current versions
echo -e "${BLUE}🔍 Getting current versions...${NC}"

FWBG_CURRENT=$(get_current_version "$FWBG_DIR")
DASHBOARD_CURRENT=$(get_current_version "$DASHBOARD_DIR")

echo -e "  fwbg:           v$FWBG_CURRENT"
echo -e "  fwbg-dashboard: v$DASHBOARD_CURRENT"

# Calculate new versions
FWBG_NEW=$(calculate_new_version "$FWBG_CURRENT" "$VERSION_TYPE")
DASHBOARD_NEW=$(calculate_new_version "$DASHBOARD_CURRENT" "$VERSION_TYPE")

echo -e "\n${YELLOW}📦 New versions:${NC}"
echo -e "  fwbg:           v$FWBG_CURRENT → v$FWBG_NEW"
echo -e "  fwbg-dashboard: v$DASHBOARD_CURRENT → v$DASHBOARD_NEW"

# Confirm
read -p "Continue with release? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Aborted.${NC}"
    exit 1
fi

# Update dashboard package.json version
if [[ -f "$DASHBOARD_DIR/package.json" ]]; then
    echo -e "\n${BLUE}📝 Updating fwbg-dashboard package.json...${NC}"
    cd "$DASHBOARD_DIR"
    # Use node to update package.json
    node -e "
        const fs = require('fs');
        const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
        pkg.version = '$DASHBOARD_NEW';
        fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2) + '\n');
    "
    echo -e "${GREEN}✅ Updated package.json to v$DASHBOARD_NEW${NC}"
fi

# Release both projects
release_project "fwbg" "$FWBG_DIR" "$FWBG_NEW"
release_project "fwbg-dashboard" "$DASHBOARD_DIR" "$DASHBOARD_NEW"

echo -e "\n${GREEN}🎉 Release complete!${NC}"
echo -e "${BLUE}📋 GitHub Actions will now build and publish the Docker images.${NC}"
echo -e "\nMonitor builds at:"
echo -e "  fwbg:           https://github.com/haexhub/fwbg/actions"
echo -e "  fwbg-dashboard: https://github.com/haexhub/fwbg-dashboard/actions"
