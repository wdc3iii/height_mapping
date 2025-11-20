#!/bin/bash
set -e

# 1) Make sure ~/.bash_aliases is sourced from ~/.bashrc
block='if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi'
if ! grep -q "$block" ~/.bashrc; then
    echo "$block" >> ~/.bashrc
    echo "Added sourcing of ~/.bash_aliases to ~/.bashrc"
fi

# 2) Ensure ~/.bash_aliases exists
if [ ! -f ~/.bash_aliases ]; then
    touch ~/.bash_aliases
    echo "Created ~/.bash_aliases"
fi

# 3) Manage a dedicated block for height_mapping
height_block=$(cat << 'EOF'
# >>> height_mapping >>>
# !! Managed by height_mapping setup script !!

alias height_mapping='source /opt/ros/humble/setup.bash && source ~/ws_livox/install/setup.bash && source ~/repos/height_mapping/install/setup.bash'
alias livox='source /opt/ros/humble/setup.bash && source ~/ws_livox/install/setup.bash'

# <<< height_mapping <<<
EOF
)

# 4) Remove any existing block and append the new one
sed '/# >>> height_mapping >>>/,/# <<< height_mapping <<</d' ~/.bash_aliases > ~/.bash_aliases.tmp \
    && mv ~/.bash_aliases.tmp ~/.bash_aliases

echo "$height_block" >> ~/.bash_aliases
echo "height_mapping aliases added to ~/.bash_aliases"

# 5) Ensure ROS_DOMAIN_ID=2 is present in ~/.bashrc
ros_line='export ROS_DOMAIN_ID=2'
if ! grep -qF "$ros_line" ~/.bashrc; then
    echo "$ros_line" >> ~/.bashrc
    echo "Set ROS_DOMAIN_ID=2 in ~/.bashrc"
fi

source ~/.bashrc