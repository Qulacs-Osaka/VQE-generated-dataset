OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.7567233431893836) q[0];
rz(-0.3308591413750444) q[0];
ry(-2.805924282116928) q[1];
rz(-1.3030611965179695) q[1];
ry(0.9408794679259805) q[2];
rz(-1.6169295731181579) q[2];
ry(2.134189950262125) q[3];
rz(-2.186096416801819) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.927602997473692) q[0];
rz(-3.070217009967211) q[0];
ry(-3.105487293280382) q[1];
rz(-2.7720142019420804) q[1];
ry(0.8021346886357374) q[2];
rz(-1.9801822809444134) q[2];
ry(3.082413599152851) q[3];
rz(-1.8805410451439837) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.9412191243497874) q[0];
rz(2.956227863855715) q[0];
ry(1.1563306603565646) q[1];
rz(2.170215278390514) q[1];
ry(-1.5322213609038657) q[2];
rz(2.006671541473243) q[2];
ry(1.3679508082510576) q[3];
rz(1.4191746282866429) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.335284198822713) q[0];
rz(-1.2087923712582311) q[0];
ry(2.1349687992824604) q[1];
rz(-0.6831744926771645) q[1];
ry(0.628226946784415) q[2];
rz(-2.844652976434037) q[2];
ry(3.0051799856994195) q[3];
rz(-2.596959930491934) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.9892107147904481) q[0];
rz(2.3543242466463496) q[0];
ry(2.495633968735021) q[1];
rz(-1.689968186825221) q[1];
ry(-2.6692992667991597) q[2];
rz(-2.737549543720189) q[2];
ry(2.090776314995394) q[3];
rz(-1.4405982656582546) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.048710629303490904) q[0];
rz(0.5364291491417488) q[0];
ry(-2.7950022388286753) q[1];
rz(1.9470227421037254) q[1];
ry(0.7098727266103069) q[2];
rz(2.2359623559182813) q[2];
ry(2.663679051582602) q[3];
rz(0.7065385684339587) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.8120160387329607) q[0];
rz(-1.8397951411891906) q[0];
ry(-0.628220509193687) q[1];
rz(0.5687697321398506) q[1];
ry(0.5523878698297202) q[2];
rz(-1.1643559085575566) q[2];
ry(-2.5725389782003103) q[3];
rz(-1.3629178954173096) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.030406009385765) q[0];
rz(-1.6279633106997735) q[0];
ry(-3.0014079999672956) q[1];
rz(-2.2090369628307593) q[1];
ry(-0.9202498887165367) q[2];
rz(1.9638629407907398) q[2];
ry(1.3830413505253674) q[3];
rz(0.5169262837707652) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.2248611611508133) q[0];
rz(1.2834706999874188) q[0];
ry(-3.1053193942856443) q[1];
rz(2.8828523755702267) q[1];
ry(-2.427565103259013) q[2];
rz(2.9184467591664895) q[2];
ry(-1.225135269939214) q[3];
rz(-1.2076914105495433) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.2208272894082635) q[0];
rz(-0.44947339873566877) q[0];
ry(2.3311548870466114) q[1];
rz(-0.38481750165320255) q[1];
ry(0.7615220627928077) q[2];
rz(-0.6565465033944542) q[2];
ry(0.0822520307535885) q[3];
rz(-0.5574843128466416) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-3.0427608479309423) q[0];
rz(-2.7310571936046864) q[0];
ry(2.6838480955672814) q[1];
rz(2.857203247058274) q[1];
ry(2.267979440041629) q[2];
rz(2.608737749240496) q[2];
ry(0.14203012202063264) q[3];
rz(0.7852416500706694) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.0567640823757678) q[0];
rz(-0.7029624965900592) q[0];
ry(0.49183813220834516) q[1];
rz(2.248234219370971) q[1];
ry(-0.815839301453436) q[2];
rz(-2.0512277662861367) q[2];
ry(0.13510351334011514) q[3];
rz(-0.28099068373880415) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.1475431622917478) q[0];
rz(-0.6529724876893398) q[0];
ry(1.6550444096490806) q[1];
rz(-1.715557182702259) q[1];
ry(3.0100190218955136) q[2];
rz(-1.2936930596853333) q[2];
ry(2.7431040371738633) q[3];
rz(1.9276352078559742) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.900194534752294) q[0];
rz(3.0796618909058693) q[0];
ry(-2.603153566979399) q[1];
rz(0.44785155456272463) q[1];
ry(2.818751375054931) q[2];
rz(-1.797060492502902) q[2];
ry(-1.9012542840581297) q[3];
rz(-0.35411906235133195) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.1980156573890466) q[0];
rz(2.7395024902311462) q[0];
ry(0.7615174715703792) q[1];
rz(1.9121243870027431) q[1];
ry(2.0964775591437714) q[2];
rz(2.395183216110207) q[2];
ry(2.183281874160628) q[3];
rz(-2.6280138233640664) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.127528145255137) q[0];
rz(-1.456545250632882) q[0];
ry(-0.6390774510406798) q[1];
rz(2.924840432418741) q[1];
ry(-2.9321377302983254) q[2];
rz(2.734368341535036) q[2];
ry(-1.600107237351854) q[3];
rz(2.3438589929583107) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.693565460703529) q[0];
rz(2.5365574404278237) q[0];
ry(0.6929319043123411) q[1];
rz(-2.0264689237241367) q[1];
ry(-1.9424117963174712) q[2];
rz(0.9768250354514701) q[2];
ry(-2.124764494688362) q[3];
rz(2.2913604036682624) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.303251822080339) q[0];
rz(1.0280093921073463) q[0];
ry(-2.203686934948728) q[1];
rz(0.9052136097790624) q[1];
ry(0.8909061474639062) q[2];
rz(-1.4435183284191135) q[2];
ry(-0.5046624067997934) q[3];
rz(-0.22501745829641703) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.7785961014951783) q[0];
rz(0.2888105365577821) q[0];
ry(1.2604830363660753) q[1];
rz(1.162555209767213) q[1];
ry(-0.31002272696752875) q[2];
rz(-2.4269570998147727) q[2];
ry(0.4022195553008304) q[3];
rz(2.107659468237759) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.3683441196163821) q[0];
rz(0.9823845825785045) q[0];
ry(1.9043843608551727) q[1];
rz(2.9674622227895706) q[1];
ry(-0.6206895762950885) q[2];
rz(2.9852240891066226) q[2];
ry(1.9336173460553567) q[3];
rz(1.877318336959194) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.0006681913561537) q[0];
rz(2.3928548141550285) q[0];
ry(1.2271285586030884) q[1];
rz(-2.457838784169274) q[1];
ry(0.9542037035326937) q[2];
rz(-2.9264674483682125) q[2];
ry(0.751741398986028) q[3];
rz(1.4159975113882677) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.2915085928609606) q[0];
rz(1.9792248990851808) q[0];
ry(-1.440792450823912) q[1];
rz(-2.0263799552874584) q[1];
ry(1.2217696640358104) q[2];
rz(-0.3261806959001547) q[2];
ry(-0.06958191599409158) q[3];
rz(2.4781126993480576) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.0998029642120717) q[0];
rz(0.7140003782610852) q[0];
ry(2.6175222868298893) q[1];
rz(0.1928879724016008) q[1];
ry(-2.2142841963737094) q[2];
rz(0.7937540827473047) q[2];
ry(-1.0188163552320313) q[3];
rz(1.7122129110432662) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.7652450138482445) q[0];
rz(-2.0431310996144303) q[0];
ry(2.2932938608345137) q[1];
rz(-1.632992567812399) q[1];
ry(2.0983394962853996) q[2];
rz(-2.3394887319770428) q[2];
ry(1.0986142838417887) q[3];
rz(-2.606202557736809) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.9357714361985696) q[0];
rz(1.344970981996262) q[0];
ry(-2.8339225310471168) q[1];
rz(-1.6677554886660644) q[1];
ry(1.0358116473868255) q[2];
rz(-3.122862085790272) q[2];
ry(-1.1617788383383898) q[3];
rz(-0.6116272493211355) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.458876123045534) q[0];
rz(1.3037432349259102) q[0];
ry(2.1971662879479723) q[1];
rz(-0.8057375793560277) q[1];
ry(2.5738525252739466) q[2];
rz(-2.9770440175736472) q[2];
ry(-1.9528913615038794) q[3];
rz(-1.3242005611168857) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.21414069786401) q[0];
rz(-0.7766244639300454) q[0];
ry(-2.7430017850977437) q[1];
rz(-0.4189610595331403) q[1];
ry(3.1371739420782205) q[2];
rz(-2.600054651808813) q[2];
ry(-2.372586450832801) q[3];
rz(2.5005377956890458) q[3];