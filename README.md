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

The slides can be accessed [here](https://docs.google.com/presentation/d/15e-hDpB42bxH5XxotDdTEYCQuPMuuE-UuLfT4cRI11A/edit?usp=sharing).

For a reference of all the geometric primitives, domain deformations, and CSG operations covered, see [iq's site](https://iquilezles.org/www/articles/distfunctions/distfunctions.htm).
