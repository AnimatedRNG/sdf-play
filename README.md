# sdf-play

If you're here for my Splash class (C12710), please run the following commands

```
sudo apt-get install cargo
git clone -b fixes-for-afs https://github.com/AnimatedRNG/sdf-play.git
cd sdf-play
cargo run
```

(when prompted for a password, use the password written on the board)

_`sdf-play` automatically reloads your code by using a library that watches files for changes. It looks like that library is not working reliably with AFS, the file system on Athena. This branch of the repository just repeatedly checks the files every second for changes and recompiles as necessary rather than relying on the file system watch._
