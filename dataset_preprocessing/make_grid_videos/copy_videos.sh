# Define remote server details
REMOTE_USER=pghosh
REMOTE_HOST='brown.is.localnet'
#REMOTE_DIR='/is/cluster/fast/pghosh/ouputs/video_gan_runs/ten_motions/00086-ffhq-ffhq_X_10_good_motions_10_motions-gpus8-batch32-gamma1/video/'
REMOTE_DIR='/is/cluster/fast/pghosh/datasets/ffhqXcelebVhq_firstorder_motion_model/ffhq_X_celebv_hq'

# Ensure there's no newline or trailing slash which might affect the command
REMOTE_DIR=$(echo $REMOTE_DIR | xargs)

# Define local directory to save files
#LOCAL_DIR="/home/pghosh/Downloads/generated_videos/talking_faces/"
LOCAL_DIR="/home/pghosh/Downloads/generated_videos/talking_faces/real"

# Ensure the local directory exists
mkdir -p "$LOCAL_DIR"

# Use 'find' to get a list of the first 32 video files and 'xargs' to handle them properly
FILES=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "find ${REMOTE_DIR} -maxdepth 1 -name '*.mp4' | head -32")

# Loop through each file and copy it to the local directory
for FILE in $FILES; do
  scp "${REMOTE_USER}@${REMOTE_HOST}:${FILE}" "${LOCAL_DIR}"
done
