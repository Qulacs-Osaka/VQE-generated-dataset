OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.4451065244644468) q[0];
ry(-2.7279256343516365) q[1];
cx q[0],q[1];
ry(0.29994341935047153) q[0];
ry(2.429649143587941) q[1];
cx q[0],q[1];
ry(-1.6472995288531886) q[1];
ry(-0.2519986727720411) q[2];
cx q[1],q[2];
ry(-1.706482650529936) q[1];
ry(-1.0943538650242148) q[2];
cx q[1],q[2];
ry(1.3206932929712742) q[2];
ry(3.1193632629523855) q[3];
cx q[2],q[3];
ry(2.6548375491052485) q[2];
ry(0.6662118176332443) q[3];
cx q[2],q[3];
ry(1.6470750344611824) q[0];
ry(0.10129926176167993) q[1];
cx q[0],q[1];
ry(0.2104106281253687) q[0];
ry(-1.1286904244377114) q[1];
cx q[0],q[1];
ry(-1.295418484345852) q[1];
ry(1.020865894861796) q[2];
cx q[1],q[2];
ry(3.0221313995774874) q[1];
ry(-3.1350705662667377) q[2];
cx q[1],q[2];
ry(2.294818872461326) q[2];
ry(-0.7373715559126943) q[3];
cx q[2],q[3];
ry(0.3484578118483368) q[2];
ry(0.08295842331134563) q[3];
cx q[2],q[3];
ry(-1.048293567329475) q[0];
ry(-2.9104000113775617) q[1];
cx q[0],q[1];
ry(1.704599017313524) q[0];
ry(2.025354382581773) q[1];
cx q[0],q[1];
ry(1.8299395325515553) q[1];
ry(1.2656232188111467) q[2];
cx q[1],q[2];
ry(-2.3719603177594255) q[1];
ry(0.6904098358602857) q[2];
cx q[1],q[2];
ry(-1.6592988951637928) q[2];
ry(1.8437747707615508) q[3];
cx q[2],q[3];
ry(2.1292306089118123) q[2];
ry(-1.4749234905598163) q[3];
cx q[2],q[3];
ry(-2.6525375830910805) q[0];
ry(-2.7524181380532906) q[1];
cx q[0],q[1];
ry(-3.119851741036506) q[0];
ry(-2.8193378827251494) q[1];
cx q[0],q[1];
ry(1.539853274042593) q[1];
ry(-0.13113003862767234) q[2];
cx q[1],q[2];
ry(2.0090727540182085) q[1];
ry(-0.9102642558799882) q[2];
cx q[1],q[2];
ry(-1.2435306681602443) q[2];
ry(2.829312236682228) q[3];
cx q[2],q[3];
ry(-2.5049516443523543) q[2];
ry(-0.8467981603821294) q[3];
cx q[2],q[3];
ry(2.057599141406098) q[0];
ry(-0.9499163186728564) q[1];
cx q[0],q[1];
ry(-3.091360192164834) q[0];
ry(-1.7335092125948517) q[1];
cx q[0],q[1];
ry(-0.1807219239560256) q[1];
ry(-0.5782089081739931) q[2];
cx q[1],q[2];
ry(-2.7007767115621655) q[1];
ry(2.400323242864591) q[2];
cx q[1],q[2];
ry(0.004552931255832586) q[2];
ry(0.7560185297019907) q[3];
cx q[2],q[3];
ry(0.3291106256883749) q[2];
ry(-2.6677044163042862) q[3];
cx q[2],q[3];
ry(1.3509246655877352) q[0];
ry(0.34278281912234015) q[1];
cx q[0],q[1];
ry(2.460015160868006) q[0];
ry(-1.171222513226088) q[1];
cx q[0],q[1];
ry(1.9164192053361138) q[1];
ry(1.6243216163180982) q[2];
cx q[1],q[2];
ry(-1.7756309378468325) q[1];
ry(-2.5905496387245037) q[2];
cx q[1],q[2];
ry(-0.24037302257627854) q[2];
ry(-2.726975661176173) q[3];
cx q[2],q[3];
ry(2.6851254750422706) q[2];
ry(0.3421811316823681) q[3];
cx q[2],q[3];
ry(-1.689337015356508) q[0];
ry(-1.7648584156337819) q[1];
cx q[0],q[1];
ry(-0.9711881206702141) q[0];
ry(0.2559343897769102) q[1];
cx q[0],q[1];
ry(0.3414850417759752) q[1];
ry(1.0585042446546387) q[2];
cx q[1],q[2];
ry(-0.43834816889950323) q[1];
ry(0.03976432283402077) q[2];
cx q[1],q[2];
ry(2.2616464220934898) q[2];
ry(0.45220703756287567) q[3];
cx q[2],q[3];
ry(1.9176542577508018) q[2];
ry(1.47565265256035) q[3];
cx q[2],q[3];
ry(1.642405523084407) q[0];
ry(-2.3999046482562894) q[1];
cx q[0],q[1];
ry(2.7545729795737857) q[0];
ry(-2.8892793065521074) q[1];
cx q[0],q[1];
ry(2.8594260973901178) q[1];
ry(2.6326416955819303) q[2];
cx q[1],q[2];
ry(-1.7320171433493812) q[1];
ry(0.31991469650357907) q[2];
cx q[1],q[2];
ry(1.577427695222782) q[2];
ry(-1.4099411387190894) q[3];
cx q[2],q[3];
ry(-1.7738771623523082) q[2];
ry(2.9059381089308993) q[3];
cx q[2],q[3];
ry(1.3007094232046192) q[0];
ry(0.5691058223416023) q[1];
cx q[0],q[1];
ry(-2.5666501169334475) q[0];
ry(0.2845815211912397) q[1];
cx q[0],q[1];
ry(2.7672546203380843) q[1];
ry(0.03240087097233139) q[2];
cx q[1],q[2];
ry(2.4165206853184835) q[1];
ry(2.24616841117558) q[2];
cx q[1],q[2];
ry(0.8435460419614058) q[2];
ry(-0.2774716243162629) q[3];
cx q[2],q[3];
ry(1.1902643081200812) q[2];
ry(3.0532485805690444) q[3];
cx q[2],q[3];
ry(3.1246169019745165) q[0];
ry(1.751878808888109) q[1];
cx q[0],q[1];
ry(0.958272612026369) q[0];
ry(-1.5682934148079342) q[1];
cx q[0],q[1];
ry(-2.22811241565065) q[1];
ry(-2.5404058292270735) q[2];
cx q[1],q[2];
ry(2.571465312754965) q[1];
ry(-0.5228858011175177) q[2];
cx q[1],q[2];
ry(1.7753437178625582) q[2];
ry(-0.9244915595907792) q[3];
cx q[2],q[3];
ry(1.5177474319743083) q[2];
ry(-2.691741730959159) q[3];
cx q[2],q[3];
ry(1.9476312363479835) q[0];
ry(-1.841920023497262) q[1];
cx q[0],q[1];
ry(2.5823125897158343) q[0];
ry(2.0870994224214092) q[1];
cx q[0],q[1];
ry(0.39064508375688817) q[1];
ry(2.8333758922332195) q[2];
cx q[1],q[2];
ry(-0.11677714780656497) q[1];
ry(1.4849959639403432) q[2];
cx q[1],q[2];
ry(-1.0532088998866413) q[2];
ry(-2.808865203489549) q[3];
cx q[2],q[3];
ry(-2.232074903044555) q[2];
ry(-0.5244416282016652) q[3];
cx q[2],q[3];
ry(1.382614907297314) q[0];
ry(1.3866051033195763) q[1];
cx q[0],q[1];
ry(3.0736907060038097) q[0];
ry(-1.579886108540011) q[1];
cx q[0],q[1];
ry(1.123250703110186) q[1];
ry(-1.8950523319632793) q[2];
cx q[1],q[2];
ry(-2.4237731954632133) q[1];
ry(1.8042425032570488) q[2];
cx q[1],q[2];
ry(1.053429786963414) q[2];
ry(1.0827389151725173) q[3];
cx q[2],q[3];
ry(-2.3701578737566353) q[2];
ry(-0.1323039343901664) q[3];
cx q[2],q[3];
ry(0.6329257221115966) q[0];
ry(0.04654628247766573) q[1];
ry(0.9916327300470814) q[2];
ry(1.8837513054145096) q[3];