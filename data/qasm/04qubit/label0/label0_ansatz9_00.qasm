OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.003576666861722) q[0];
ry(2.5579535392859256) q[1];
cx q[0],q[1];
ry(1.0547342490115792) q[0];
ry(-2.2391897809816714) q[1];
cx q[0],q[1];
ry(-2.836517695564872) q[2];
ry(1.2893150205015176) q[3];
cx q[2],q[3];
ry(1.3132885800040714) q[2];
ry(0.8537128144769164) q[3];
cx q[2],q[3];
ry(-1.2756470550032266) q[0];
ry(1.0742022285385258) q[2];
cx q[0],q[2];
ry(-0.6280165982352031) q[0];
ry(-2.6922054111233202) q[2];
cx q[0],q[2];
ry(-0.4928050431458005) q[1];
ry(-1.8967764028710201) q[3];
cx q[1],q[3];
ry(0.3238221955994613) q[1];
ry(1.069469073672731) q[3];
cx q[1],q[3];
ry(-2.738441041453044) q[0];
ry(-1.1641252536098108) q[3];
cx q[0],q[3];
ry(2.0002510129341604) q[0];
ry(-2.107888390126549) q[3];
cx q[0],q[3];
ry(1.923762636580148) q[1];
ry(1.7721068011859789) q[2];
cx q[1],q[2];
ry(0.2846274671154462) q[1];
ry(2.4003249284599963) q[2];
cx q[1],q[2];
ry(1.9797479566119294) q[0];
ry(2.0263102898562964) q[1];
cx q[0],q[1];
ry(1.255988527593085) q[0];
ry(-1.4323783085675432) q[1];
cx q[0],q[1];
ry(-0.1030395568088256) q[2];
ry(0.901774463712262) q[3];
cx q[2],q[3];
ry(0.4888992814530351) q[2];
ry(-0.720552141797046) q[3];
cx q[2],q[3];
ry(-1.158769123995048) q[0];
ry(0.43450947325544487) q[2];
cx q[0],q[2];
ry(1.7479982580907327) q[0];
ry(2.917906155869543) q[2];
cx q[0],q[2];
ry(0.4587743976545291) q[1];
ry(-0.25645079125554915) q[3];
cx q[1],q[3];
ry(2.535972735869743) q[1];
ry(0.9090833939869789) q[3];
cx q[1],q[3];
ry(2.974343300781054) q[0];
ry(3.0367106387860425) q[3];
cx q[0],q[3];
ry(1.1832883536505145) q[0];
ry(-2.808790421872024) q[3];
cx q[0],q[3];
ry(-0.48343984088956965) q[1];
ry(-1.7065211363164956) q[2];
cx q[1],q[2];
ry(-0.37467262260245443) q[1];
ry(-0.6161775639544276) q[2];
cx q[1],q[2];
ry(2.532531510471629) q[0];
ry(-2.4519774536964785) q[1];
cx q[0],q[1];
ry(0.6263003957709046) q[0];
ry(0.7121482284355672) q[1];
cx q[0],q[1];
ry(0.01944801570660459) q[2];
ry(-0.5110561786222788) q[3];
cx q[2],q[3];
ry(0.7156822798585498) q[2];
ry(-0.9338356177589056) q[3];
cx q[2],q[3];
ry(2.2042973401039463) q[0];
ry(-1.1718191025575138) q[2];
cx q[0],q[2];
ry(-1.31746916129004) q[0];
ry(-0.8693226901236725) q[2];
cx q[0],q[2];
ry(0.8533462769084065) q[1];
ry(1.352218811344728) q[3];
cx q[1],q[3];
ry(2.905249304483513) q[1];
ry(-2.8793418741926216) q[3];
cx q[1],q[3];
ry(-2.401054657624353) q[0];
ry(2.721738985606052) q[3];
cx q[0],q[3];
ry(-0.12268063415458563) q[0];
ry(-2.7668690023306373) q[3];
cx q[0],q[3];
ry(1.577121702716653) q[1];
ry(-0.5607161435650205) q[2];
cx q[1],q[2];
ry(2.0572762137702933) q[1];
ry(2.7936444382731245) q[2];
cx q[1],q[2];
ry(2.7304143917584924) q[0];
ry(0.6181823392751907) q[1];
ry(-1.4806553369775939) q[2];
ry(1.222238157624026) q[3];