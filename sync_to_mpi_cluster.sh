dir_to_sync='eg3d'
while inotifywait -r ../$dir_to_sync/*; do
  rsync -azhe "ssh -i ~/.ssh/id_rsa" ../$dir_to_sync/ pghosh@login.cluster.is.localnet:/home/pghosh/repos/never_eidt_from_cluster_remote_edit_loc/$dir_to_sync
done
