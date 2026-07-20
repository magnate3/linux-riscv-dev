# ChaCha implementation on Tofino switches
This is a P4 implementation of ChaCha on Intel Tofino programmable switches. The original paper proposing this implementation has been accepted to appear in Euro P4 workshop, which will be held in conjunction with CoNEXT 2022. 

## Running on tofino_model
- `$ cd bf-p4c -g chacha.p4 -o $OUT_DIR`
- `$ ./run_tofino_model.sh -c $OUT_DIR/chacha.conf -p chacha`
- `$ ./run_switchd.sh -c $OUT_DIR/chacha.conf -p chacha`
- `$ python3 control.py`
- `$ sudo python3 test.py`

## Contributors
- Yutaro Yoshinaka (@io-sink) (Graduate School of Information Science and Technology, Osaka University)
- Junji Takemasa (@j-takemasa) (Graduate School of Information Science and Technology, Osaka University)
- Yuki Koizumi (@yuki-koizumi) (Graduate School of Information Science and Technology, Osaka University)
- Toru Hasegawa (Graduate School of Information Science and Technology, Osaka University)

## License
This program is released under the [GNU Affero General Public License v3](https://www.gnu.org/licenses/agpl-3.0.html).

## Reference
1. Y. Yoshinaka, J. Takemasa, Y. Koizumi, and T. Hasegawa, “On Implementing ChaCha on a Programmable Switch” to appear in Proceedings of European P4 Workshop (EuroP4), 2022. 