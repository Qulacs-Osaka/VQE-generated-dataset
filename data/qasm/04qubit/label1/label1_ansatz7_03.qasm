OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.4217768527708978) q[0];
ry(2.6349584041532395) q[1];
cx q[0],q[1];
ry(-2.868232631586247) q[0];
ry(-2.4169137525609377) q[1];
cx q[0],q[1];
ry(0.006570970770708782) q[0];
ry(-2.1587172208875414) q[2];
cx q[0],q[2];
ry(-2.7061459102068537) q[0];
ry(0.7645662591838212) q[2];
cx q[0],q[2];
ry(0.35879477877271615) q[0];
ry(0.995146703256637) q[3];
cx q[0],q[3];
ry(-1.1866002528612931) q[0];
ry(0.4002693429669765) q[3];
cx q[0],q[3];
ry(0.4198528277074471) q[1];
ry(-2.7858151873105403) q[2];
cx q[1],q[2];
ry(-1.8929292362548091) q[1];
ry(1.386015156928013) q[2];
cx q[1],q[2];
ry(-2.7253318768529495) q[1];
ry(-1.698094070998253) q[3];
cx q[1],q[3];
ry(-2.7230504472751615) q[1];
ry(1.1943958709139608) q[3];
cx q[1],q[3];
ry(2.837657050833874) q[2];
ry(-3.074705012028617) q[3];
cx q[2],q[3];
ry(0.9406271863457832) q[2];
ry(-0.9678621721902011) q[3];
cx q[2],q[3];
ry(0.055431669029325496) q[0];
ry(2.375338339595737) q[1];
cx q[0],q[1];
ry(1.3341706348308366) q[0];
ry(2.267220512131833) q[1];
cx q[0],q[1];
ry(-2.591228975803443) q[0];
ry(-0.6777374386498333) q[2];
cx q[0],q[2];
ry(0.6563327512961262) q[0];
ry(1.7053413967839919) q[2];
cx q[0],q[2];
ry(-0.4958897191617089) q[0];
ry(0.14153777503089413) q[3];
cx q[0],q[3];
ry(-2.782774556700888) q[0];
ry(-0.8287945101980846) q[3];
cx q[0],q[3];
ry(-2.53169595341393) q[1];
ry(-0.9349654417822668) q[2];
cx q[1],q[2];
ry(-2.3740417432578758) q[1];
ry(-1.7572810067619962) q[2];
cx q[1],q[2];
ry(-0.09756671329038083) q[1];
ry(3.0645070797124356) q[3];
cx q[1],q[3];
ry(2.9532856177155766) q[1];
ry(0.7468050800876562) q[3];
cx q[1],q[3];
ry(1.310805389710029) q[2];
ry(-1.618175851996119) q[3];
cx q[2],q[3];
ry(-2.918175367363031) q[2];
ry(0.8310871748741846) q[3];
cx q[2],q[3];
ry(-0.0809897450431798) q[0];
ry(2.4563964733953316) q[1];
cx q[0],q[1];
ry(2.138788722621452) q[0];
ry(-1.135211970990233) q[1];
cx q[0],q[1];
ry(-2.496281650138571) q[0];
ry(-0.04554793508218427) q[2];
cx q[0],q[2];
ry(-0.5945958026539068) q[0];
ry(1.2274701069636815) q[2];
cx q[0],q[2];
ry(-2.632057434149142) q[0];
ry(-0.18346301367876566) q[3];
cx q[0],q[3];
ry(-0.93275192881545) q[0];
ry(-1.9136585426551846) q[3];
cx q[0],q[3];
ry(-2.8921589058617085) q[1];
ry(-2.806102571094316) q[2];
cx q[1],q[2];
ry(2.220126301283597) q[1];
ry(2.382527820125243) q[2];
cx q[1],q[2];
ry(0.9935626661232082) q[1];
ry(-2.4447302485493756) q[3];
cx q[1],q[3];
ry(-1.2917851539235845) q[1];
ry(1.0234068493262942) q[3];
cx q[1],q[3];
ry(2.2407904867481543) q[2];
ry(0.4196626423848038) q[3];
cx q[2],q[3];
ry(1.9142973380136106) q[2];
ry(-2.960624390590177) q[3];
cx q[2],q[3];
ry(-1.8401072683804758) q[0];
ry(1.7808852185159234) q[1];
cx q[0],q[1];
ry(1.2786412714510416) q[0];
ry(2.7219727958293474) q[1];
cx q[0],q[1];
ry(-2.7210279540330973) q[0];
ry(-0.38244802601804756) q[2];
cx q[0],q[2];
ry(-2.0257729312334556) q[0];
ry(2.848893072164886) q[2];
cx q[0],q[2];
ry(0.7605621831494105) q[0];
ry(2.126266786919089) q[3];
cx q[0],q[3];
ry(-0.5195046529628773) q[0];
ry(-0.34500547352640903) q[3];
cx q[0],q[3];
ry(1.1129247414806462) q[1];
ry(2.586216008013262) q[2];
cx q[1],q[2];
ry(-0.4839518419882501) q[1];
ry(2.4217067034940296) q[2];
cx q[1],q[2];
ry(-0.6016144210715151) q[1];
ry(-0.6955710767643906) q[3];
cx q[1],q[3];
ry(-0.9385782992852647) q[1];
ry(2.2171459815769747) q[3];
cx q[1],q[3];
ry(-1.5643467186231665) q[2];
ry(-3.115363007463315) q[3];
cx q[2],q[3];
ry(-0.8162770873105517) q[2];
ry(1.3256286614879471) q[3];
cx q[2],q[3];
ry(-2.685357890031625) q[0];
ry(-0.4154078309999969) q[1];
cx q[0],q[1];
ry(1.5224511615576688) q[0];
ry(-0.9394849430133474) q[1];
cx q[0],q[1];
ry(-0.4299744324171862) q[0];
ry(-0.144013856578737) q[2];
cx q[0],q[2];
ry(2.314269747225668) q[0];
ry(-1.6671207385871445) q[2];
cx q[0],q[2];
ry(-0.1949046666597639) q[0];
ry(2.3743053754392998) q[3];
cx q[0],q[3];
ry(1.0134837172825302) q[0];
ry(-0.3062796869819265) q[3];
cx q[0],q[3];
ry(-1.7513158452179012) q[1];
ry(-1.1159832605420625) q[2];
cx q[1],q[2];
ry(2.3240837699179964) q[1];
ry(0.804633149269986) q[2];
cx q[1],q[2];
ry(-1.3547336469824403) q[1];
ry(-2.590012506124783) q[3];
cx q[1],q[3];
ry(1.3945669286297857) q[1];
ry(-2.9857030596232454) q[3];
cx q[1],q[3];
ry(2.9027700509588703) q[2];
ry(0.7602964782111226) q[3];
cx q[2],q[3];
ry(1.0405826445753767) q[2];
ry(-0.5119878484597505) q[3];
cx q[2],q[3];
ry(-1.195184557985161) q[0];
ry(-2.459655554406087) q[1];
cx q[0],q[1];
ry(2.2495116583479584) q[0];
ry(-1.3851360877903005) q[1];
cx q[0],q[1];
ry(2.8044302910443255) q[0];
ry(1.0391915639124312) q[2];
cx q[0],q[2];
ry(-0.6609357208358979) q[0];
ry(-2.2021950893052944) q[2];
cx q[0],q[2];
ry(0.9566368865500142) q[0];
ry(-0.8186749501056454) q[3];
cx q[0],q[3];
ry(-1.3612303154347811) q[0];
ry(-1.574949671446154) q[3];
cx q[0],q[3];
ry(0.35657662292960074) q[1];
ry(1.8869539097776504) q[2];
cx q[1],q[2];
ry(1.914036726587032) q[1];
ry(2.0361833766716195) q[2];
cx q[1],q[2];
ry(-1.8275497364936717) q[1];
ry(-2.6217101277296924) q[3];
cx q[1],q[3];
ry(0.09803744378301077) q[1];
ry(2.724882511857829) q[3];
cx q[1],q[3];
ry(-2.0125147426336367) q[2];
ry(-1.9201935905726344) q[3];
cx q[2],q[3];
ry(-1.1246296831628548) q[2];
ry(-1.1277229021357922) q[3];
cx q[2],q[3];
ry(1.420769648256523) q[0];
ry(-2.3616982635506933) q[1];
ry(-3.0899461002047035) q[2];
ry(-0.6685660521625854) q[3];