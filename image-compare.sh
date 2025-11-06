#!/opt/homebrew/bin/bash

images=0
image_attempts=0
cutoff=200

template=$1
account=$2

refresh_account() {
  (( image_attempts < cutoff )) && return

  # for now just return
  return

  echo "switch!"

  # looks like we should switch accounts
  cliclick -f "/Volumes/bench/Word Blender/scripts/imagefx/imagefx-signout-$(hostname).script"
  for i in $(seq ${account}); do
    cliclick -f "/Volumes/bench/Word Blender/scripts/imagefx/tab.script"
  done
  cliclick -f "/Volumes/bench/Word Blender/scripts/imagefx/imagefx-setup-$(hostname).script"

  image_attempts=0
  account=$(( account + 1 ))
  (( account == 3 )) && account=1
}

pull_image() {
  the_word="${1}"

  refresh_account

  cat "${template}" | sed "s/;/${the_word}/g" > /tmp/run.script
  cliclick -f /tmp/run.script || exit

  sleep 2

  # whether or not we got an image, we did run something through the AI
  images=$(( images + 1 ))
  image_attempts=$(( image_attempts + 1 ))
}


submit_image () {
    the_word="${1}"
    mv ~/Downloads/Gemini_Generated_Image_*.png /tmp/image.png

    mv /tmp/image.png samples/"${the_word}".png
    return

    if [[ ! -e /tmp/image.png ]]; then
      echo " reporting image as missing"
      echo "${the_word}" >> missing_images.txt
      return
    fi

    cmd="/usr/bin/curl -s -X POST https://wordblender.us/partner/13523688-376a-4860-a3ba-29f80c430fd9/image/checkin -F word=\"${the_word}\" -F \"image=@/tmp/image.png\""
    json="$(/usr/bin/curl -s -X POST https://wordblender.us/partner/13523688-376a-4860-a3ba-29f80c430fd9/image/checkin -F word="${the_word}" -F "image=@/tmp/image.png")"
    success="$(echo "${json}" | jq -r .success)"
    if [[ "${success}" != "true" ]]; then
        echo "Failed to check in image: ${cmd} => ${json}"
        echo "${the_word}" >> missing_images.txt
        return
    fi
    rm /tmp/image.png
}

while true; do
    # Check out a word
    json="$(/usr/bin/curl -s https://wordblender.us/partner/13523688-376a-4860-a3ba-29f80c430fd9/image/checkout)"

    #parse the json {"success":true,"word":"finger wars","deadline":"2025-01-11T04:51:40.922Z"}
    success="$(echo "${json}" | jq -r .success)"
    word="$(echo "${json}" | jq -r .word)"
    deadline="$(echo "${json}" | jq -r .deadline)"

    if [[ "${success}" != "true" ]]; then
        echo "Failed to check out word: ${json}"
        sleep 60
        continue
    fi

    echo "[${image_attempts}] Checked out word: ${word}"

    pull_image "${word}"
    submit_image "${word}"

    echo "sleeping for break time. Exit now if you'd like."
    sleep 10
done
