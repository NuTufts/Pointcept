#!/bin/bash
#
# _wconfig_common.sh
# Shared bootstrap for the _wconfig subscripts. Enables standalone invocation:
#
#   source scripts/run_step1_lantern_wconfig.sh   configs/onefile_test.conf [lineno]
#   source scripts/run_step234_pointcept_wconfig.sh configs/onefile_test.conf [lineno]
#   source scripts/run_step567_pointcept_wconfig.sh configs/onefile_test.conf [lineno]
#
# The helper auto-detects whether to enter standalone mode (arg1 is a config
# file) or driver mode (arg1 is a workdir path, existing behavior). In
# standalone mode, if the current shell is not already inside the subscript's
# target container, the bootstrap re-execs itself via `apptainer exec`.
#
# A subscript that wants standalone support adds, near the top:
#
#   WCONFIG_CONTAINER_TYPE=lantern    # or 'pointcept'
#   source "$(dirname "${BASH_SOURCE[0]}")/_wconfig_common.sh"
#   wconfig_bootstrap "$@"; _bs_rc=$?
#   if [ "${WCONFIG_RE_EXECED:-0}" = "1" ] || [ ${_bs_rc} -ne 0 ]; then
#       return ${_bs_rc} 2>/dev/null || exit ${_bs_rc}
#   fi
#   if [ "${WCONFIG_STANDALONE:-0}" = "1" ]; then
#       set -- <subscript's driver-mode positional args, synthesized from config>
#   fi
#   # ...rest of subscript, unchanged...
#

wconfig_is_config_file() {
    local f=${1:-}
    [ -n "$f" ] && [ -f "$f" ] && grep -q '^[[:space:]]*INPUTLIST=' "$f" 2>/dev/null
}

wconfig_bootstrap() {
    WCONFIG_STANDALONE=0
    WCONFIG_RE_EXECED=0

    if ! wconfig_is_config_file "${1:-}"; then
        return 0   # driver mode — subscript proceeds as before
    fi

    WCONFIG_STANDALONE=1
    WCONFIG_LINENO_OVERRIDE=${2:-}

    # Resolve repo anchors from the calling subscript's path.
    # Absolutize caller_script so the re-exec works regardless of the shell's
    # cwd after subsequent `cd ${WORKDIR_PATH}` inside the subscript.
    local caller_raw=${BASH_SOURCE[1]}
    SCRIPT_DIR=$(cd "$(dirname "${caller_raw}")" && pwd)
    REPO_ROOT=$(cd "${SCRIPT_DIR}" && pwd)
    local caller_script="${SCRIPT_DIR}/$(basename "${caller_raw}")"

    # Absolutize the config path too (in case it's relative)
    if [ -f "$1" ]; then
        WCONFIG_CONFIG_FILE=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
    else
        WCONFIG_CONFIG_FILE=$1
    fi
    UBDL_DIR=${UBDL_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/ubdl}
    POINTCEPT_DIR=${POINTCEPT_DIR:-/cluster/tufts/wongjiradlabnu/twongj01/pointcept_env/pointcept}
    export SCRIPT_DIR REPO_ROOT UBDL_DIR POINTCEPT_DIR

    # shellcheck disable=SC1090
    source "${WCONFIG_CONFIG_FILE}"

    WORKDIR_BASE=${WORKDIR_BASE:-/tmp}
    KEEP_INTERMEDIATES=${KEEP_INTERMEDIATES:-0}
    MAX_EVENTS=${MAX_EVENTS:--1}
    stride=${stride:-1}
    OFFSET=${OFFSET:-0}
    ADCNAME=${ADCNAME:-wire}
    TBFLAG=${TBFLAG:-""}
    MCC9FLAG=${MCC9FLAG:-""}
    DEVICE=${DEVICE:-cuda}
    SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

    local jobid=${SLURM_ARRAY_TASK_ID}
    local default_lineno=$(( OFFSET + stride * jobid + 1 ))
    WCONFIG_LINENO=${WCONFIG_LINENO_OVERRIDE:-${default_lineno}}
    WCONFIG_FILENO=${WCONFIG_LINENO}

    WCONFIG_WORKDIR=$(printf "%s/lantern_${TAG}_jobid%04d_line%05d" \
        "${WORKDIR_BASE}" ${jobid} ${WCONFIG_LINENO})
    mkdir -p "${WCONFIG_WORKDIR}"

    # Re-exec inside the target container if we're on the bare node.
    if [ -z "${APPTAINER_CONTAINER}${SINGULARITY_CONTAINER}" ]; then
        local target bind nv
        case "${WCONFIG_CONTAINER_TYPE:-}" in
            lantern)
                target=${LANTERN_CONTAINER}
                bind="/cluster/tufts:/cluster/tufts"
                case "${WORKDIR_BASE}" in
                    /cluster/tufts*) ;;
                    *) bind="${bind},${WORKDIR_BASE}:${WORKDIR_BASE}" ;;
                esac
                nv=""
                ;;
            pointcept)
                target=${POINTCEPT_CONTAINER}
                bind="/cluster:/cluster"
                case "${WORKDIR_BASE}" in
                    /cluster*) ;;
                    *) bind="${bind},${WORKDIR_BASE}:${WORKDIR_BASE}" ;;
                esac
                nv="--nv"
                ;;
            *)
                echo "ERROR: subscript must set WCONFIG_CONTAINER_TYPE=lantern|pointcept" >&2
                return 1
                ;;
        esac

        if ! command -v apptainer >/dev/null 2>&1; then
            echo "ERROR: apptainer not on PATH. Did you 'module load apptainer/1.2.4-suid'?" >&2
            return 1
        fi

        echo "wconfig_bootstrap: re-executing $(basename "${caller_script}") inside ${WCONFIG_CONTAINER_TYPE} container"
        echo "  config : ${WCONFIG_CONFIG_FILE}"
        echo "  lineno : ${WCONFIG_LINENO}"
        echo "  workdir: ${WCONFIG_WORKDIR}"
        apptainer exec ${nv} --bind "${bind}" "${target}" \
            bash -c "source ${caller_script} ${WCONFIG_CONFIG_FILE} ${WCONFIG_LINENO}"
        local rc=$?
        WCONFIG_RE_EXECED=1
        return ${rc}
    fi

    # We're inside the container already.
    # Only Step 1 needs the raw input ROOT copied in; Steps 2-4 / 5-7 work
    # from intermediate outputs that should already be in the workdir.
    if [ "${WCONFIG_CONTAINER_TYPE}" = "lantern" ]; then
        local inputfile ibn
        inputfile=$(sed -n "${WCONFIG_LINENO}p" "${INPUTLIST}")
        if [ -n "${inputfile}" ] && [ -f "${inputfile}" ]; then
            ibn=$(basename "${inputfile}")
            if [ ! -f "${WCONFIG_WORKDIR}/${ibn}" ]; then
                echo "wconfig_bootstrap: copying ${ibn} to ${WCONFIG_WORKDIR}"
                cp "${inputfile}" "${WCONFIG_WORKDIR}/"
            fi
        else
            echo "WARNING: line ${WCONFIG_LINENO} of ${INPUTLIST} is empty or points at a missing file" >&2
        fi
    fi

    return 0
}
