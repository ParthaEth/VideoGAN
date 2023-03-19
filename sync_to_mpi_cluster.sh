dir_to_sync='VideoGAN'
while inotifywait -r --exclude '/\.' ../$dir_to_sync/*; do
  rsync --exclude=".*" -azhe "ssh -i ~/.ssh/id_rsa" ../$dir_to_sync/ pghosh@brown.is.localnet:/is/cluster/pghosh/repos/never_eidt_from_cluster_remote_edit_loc/$dir_to_sync &
  sleep 2
  echo "second sync."
  rsync --exclude=".*" -azhe "ssh -i ~/.ssh/id_rsa" ../$dir_to_sync/ pghosh@brown.is.localnet:/is/cluster/pghosh/repos/never_eidt_from_cluster_remote_edit_loc/$dir_to_sync
done
