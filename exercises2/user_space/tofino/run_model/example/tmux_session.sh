#! /usr/bin/env bash
session="model_session"

# If the session already exists, we should attach to it, rather than try and 
# create a new one. It'll bork the existing session in some way.
tmux has-session -t $session 2>/dev/null

if [ $? != 0 ]; then
    tmux new-session  -s ${session} -d
    tmux set -t ${session} window-style 'fg=colour252,bg=colour234'
    tmux set -t ${session} window-active-style 'fg=default,bg=colour233'
    tmux set -t ${session} pane-border-status top
    tmux set -t ${session} allow-rename
    tmux set -t ${session} pane-active-border-style "bg=default,fg=colour214"
    tmux split-window -t ${session} -v
    tmux split-window -t ${session} -h
    tmux select-pane -t 0 -T "Controller or Scripts";
    tmux select-pane -t 1 -T "Tofino Model";
    tmux select-pane -t 2 -T "bfswitchd";

    function setup() {
        tmux send-keys -t "model_session.0" "cd ~/work/example; " Enter \
            "echo -e '\ec' \"\r$(cat quickstart.txt)\n\"" Enter
        tmux send-keys -t "model_session.1" \
            "cd \$SDE; reset" Enter \
            "./run_tofino_model.sh -p \\" Enter \
            "patch_panel -f ~/work/example/ports.json"
        tmux send-keys -t "model_session.2" "cd \$SDE; reset" Enter \
            './run_switchd.sh -p patch_panel'
    }

    # tmux select-layout -t ${session} even-vertical
    tmux select-pane -t model_session.0

    # Run setup with a delay to allow the pane's to fully complete loading,
    # otherwise the informational text will be interleaved with the prompt. It
    # doesn't look pretty.
    sleep 0.5; setup
    tmux -2 attach-session -d
else
    tmux attach-session -t ${session}
fi
