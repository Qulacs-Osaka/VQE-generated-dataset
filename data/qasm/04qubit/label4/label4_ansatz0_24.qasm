OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.025107605384503552) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09645175051718764) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.05012114173361118) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11336686791173245) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07429391881009816) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.046547322428957344) q[3];
cx q[2],q[3];
rx(-0.042494293292098265) q[0];
rz(-0.059293287385394786) q[0];
rx(-0.06268941257208657) q[1];
rz(-0.051936153436414785) q[1];
rx(-0.09379452844497244) q[2];
rz(-0.052036555888612716) q[2];
rx(-0.027001438209256) q[3];
rz(-0.08585585585611784) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.013002249646800247) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07923063957744854) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.1091976441921314) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09221375706659353) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.048905820285669636) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03411761701964573) q[3];
cx q[2],q[3];
rx(-0.002847849978186315) q[0];
rz(-0.01887930977873939) q[0];
rx(0.028593004979547892) q[1];
rz(-0.08345727661306196) q[1];
rx(-0.14571805050724707) q[2];
rz(-0.09982606589557802) q[2];
rx(-0.054152001317955514) q[3];
rz(0.002761018335043179) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09279081775285945) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0741496346239123) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.05465528364155976) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12443190864836007) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0061287249353871864) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09785202306388142) q[3];
cx q[2],q[3];
rx(-0.02092365382940481) q[0];
rz(-0.06964577611536986) q[0];
rx(-0.025435914540201603) q[1];
rz(-0.04174837465727872) q[1];
rx(-0.07419265866818385) q[2];
rz(-0.11561521506055566) q[2];
rx(-0.08962738159634437) q[3];
rz(0.002003449447826446) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.05613036881517512) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.057474282918750196) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.05770801616428491) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12578023637441763) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.05921587112908373) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08461462842838446) q[3];
cx q[2],q[3];
rx(-0.06657369447165554) q[0];
rz(-0.015802311170411445) q[0];
rx(0.013374352108003544) q[1];
rz(-0.06928832211386995) q[1];
rx(-0.06033481118768595) q[2];
rz(-0.10328672473250454) q[2];
rx(-0.10409421825454628) q[3];
rz(0.012257900225103044) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09140326756650162) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.017357628443926826) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.027173071981315536) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09255707460973586) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.012818346918219254) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03167041060846631) q[3];
cx q[2],q[3];
rx(-0.03129705958610664) q[0];
rz(-0.029796199230538733) q[0];
rx(-0.019252397605424017) q[1];
rz(-0.12436255581967708) q[1];
rx(-0.10005160181788023) q[2];
rz(-0.059994869045103094) q[2];
rx(-0.04078839126963063) q[3];
rz(-0.006009781833631249) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08807165896361581) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.009093508451968761) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.040034176368074124) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12504021245217284) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0145120707440808) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07060364442521494) q[3];
cx q[2],q[3];
rx(-0.0705718608337696) q[0];
rz(0.0039728348557121615) q[0];
rx(0.012143836480440292) q[1];
rz(-0.11091162214895016) q[1];
rx(-0.09736310498736718) q[2];
rz(-0.12244225131057765) q[2];
rx(-0.12270123674305075) q[3];
rz(-0.05661725839368778) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.045716062764547864) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.04683376857221228) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.019714963656407233) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1411963312127996) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.05263575042073173) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.02184448612913565) q[3];
cx q[2],q[3];
rx(-0.028810224262973957) q[0];
rz(-0.0663606654313769) q[0];
rx(0.07100801377743021) q[1];
rz(-0.048748211176632264) q[1];
rx(-0.10744688868024199) q[2];
rz(-0.10296475668130652) q[2];
rx(-0.15230869012420212) q[3];
rz(0.024805438473010845) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.05363080010165778) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.019800584962026704) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.0752230124468232) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.14427986172321552) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.044737755658108225) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.025334041425240685) q[3];
cx q[2],q[3];
rx(-0.031131808807723937) q[0];
rz(-0.06675111332636269) q[0];
rx(-0.010041416354949276) q[1];
rz(-0.10603825633174102) q[1];
rx(-0.05825073735905858) q[2];
rz(-0.10997409475286568) q[2];
rx(-0.09531470082618647) q[3];
rz(0.00935718975165458) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11049730783015936) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.0736146734050372) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.07833441191886295) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1110715675105673) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.04511979273927048) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0353773179642433) q[3];
cx q[2],q[3];
rx(-0.018707521602677617) q[0];
rz(-0.05113184251572227) q[0];
rx(-0.04210283167521031) q[1];
rz(-0.058698764435947945) q[1];
rx(-0.066175165364086) q[2];
rz(-0.04210347733184211) q[2];
rx(-0.11081535647550067) q[3];
rz(0.0060743084650424425) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12473952888023496) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.024102618436318703) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.07456185564890587) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.17192853621808613) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.019926863681593368) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.045223803240916544) q[3];
cx q[2],q[3];
rx(-0.1131172703049231) q[0];
rz(-0.05599704717609081) q[0];
rx(0.018921976111097128) q[1];
rz(-0.0841725627012539) q[1];
rx(-0.11397449652364126) q[2];
rz(-0.04943099918858183) q[2];
rx(-0.14272374289342638) q[3];
rz(-0.03961151116764241) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13208201118041984) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.007264366923977552) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.04726904639435206) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.13494709703420504) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.010080475592055058) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03467943767318843) q[3];
cx q[2],q[3];
rx(-0.043396133795160996) q[0];
rz(-0.0386274029481011) q[0];
rx(0.0017381903166643) q[1];
rz(-0.1028825467416858) q[1];
rx(-0.07473479963195258) q[2];
rz(-0.05058669520394846) q[2];
rx(-0.10904548147634294) q[3];
rz(-0.02947303781796871) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07425881733736396) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.017425801281760495) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.09014085053710005) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08833868964083678) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.037981392982840784) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.017815920600065056) q[3];
cx q[2],q[3];
rx(-0.12397969827904184) q[0];
rz(-0.06748328926114971) q[0];
rx(0.009521915744472415) q[1];
rz(-0.08198282125214815) q[1];
rx(-0.046103498679393465) q[2];
rz(-0.055537043376412194) q[2];
rx(-0.18383614704264156) q[3];
rz(-0.01013635653471228) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11007358389026257) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.027271587966563926) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.11377755167159065) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.16219790139475684) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.05426106290159683) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0011767376417912015) q[3];
cx q[2],q[3];
rx(-0.07930450210155623) q[0];
rz(-0.020310355874671473) q[0];
rx(0.039861154493267964) q[1];
rz(-0.04687611223563075) q[1];
rx(-0.09553989335513767) q[2];
rz(-0.13431875772563345) q[2];
rx(-0.14812745160396928) q[3];
rz(-0.054652910083595435) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06846555414137903) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.03220439150305219) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.10431016124490165) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09044630544574192) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.01738143744093719) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0065795518492804735) q[3];
cx q[2],q[3];
rx(-0.10370091743471922) q[0];
rz(-0.06391206910811241) q[0];
rx(-0.024112210537741437) q[1];
rz(-0.020756846050441237) q[1];
rx(-0.03759494547906905) q[2];
rz(-0.04135610016608189) q[2];
rx(-0.12609851903820662) q[3];
rz(0.03457866284573718) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.0009268625710336269) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.032831969001893196) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.07817200710755792) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12561799909385485) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.018175013186643326) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.010499557991803913) q[3];
cx q[2],q[3];
rx(-0.16356163862748468) q[0];
rz(-0.09174227840165629) q[0];
rx(-0.03179559256558641) q[1];
rz(-0.09818024628173576) q[1];
rx(-0.07923691825631533) q[2];
rz(-0.09782719010301018) q[2];
rx(-0.10601323401495422) q[3];
rz(-0.0025100054637483306) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07878475597198831) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.022217314506769045) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.01721218160725784) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11294185826701053) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.07378561093973014) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0008854722126046962) q[3];
cx q[2],q[3];
rx(-0.1246828813956066) q[0];
rz(-0.0035348838524851085) q[0];
rx(0.03253523528441588) q[1];
rz(-0.002010760002862063) q[1];
rx(-0.027875658471816413) q[2];
rz(-0.10107703189849601) q[2];
rx(-0.10904059966222239) q[3];
rz(0.04356407219480404) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08850219811998249) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.05012843842679915) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.047002855332069396) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06883172452941097) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.028098672524416626) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.018083274481484356) q[3];
cx q[2],q[3];
rx(-0.14100010935328974) q[0];
rz(-0.052056461004389126) q[0];
rx(-0.009816141757012148) q[1];
rz(-0.01493336613869615) q[1];
rx(0.01984858628770779) q[2];
rz(-0.1035310912912709) q[2];
rx(-0.16879282970206305) q[3];
rz(-0.016343892769076544) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.03751918204464261) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.0021879603624755314) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.03252268102250275) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08135813343869272) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.026325597859982515) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.05136910677457219) q[3];
cx q[2],q[3];
rx(-0.1493610878244091) q[0];
rz(-0.08547321216701552) q[0];
rx(-0.00994677801394111) q[1];
rz(-0.06807271333179177) q[1];
rx(-0.022897160768196315) q[2];
rz(-0.07235982158913815) q[2];
rx(-0.1497062619420773) q[3];
rz(-0.004924198801855076) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10918394487622024) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.006448227604870119) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.01783381803491357) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12507519700921035) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.01934855488668192) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.003176611282164858) q[3];
cx q[2],q[3];
rx(-0.16738730128124465) q[0];
rz(-0.06419311055587706) q[0];
rx(-0.028259069130830348) q[1];
rz(-0.012695308836495863) q[1];
rx(0.007821577874475842) q[2];
rz(-0.0756392033353226) q[2];
rx(-0.19488396422309845) q[3];
rz(0.02525620278763773) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1026639038806973) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.098790441873699) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.015706719102661103) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1258369458023652) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.025208488494440653) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.023628114465010435) q[3];
cx q[2],q[3];
rx(-0.20086671018261126) q[0];
rz(-0.02219276636303898) q[0];
rx(-0.024656812722452636) q[1];
rz(-0.0721664250920071) q[1];
rx(0.022163585520797162) q[2];
rz(-0.03834117670055603) q[2];
rx(-0.16540534177304408) q[3];
rz(0.03843104305613638) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11249239296460513) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09189375367855784) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.016445618097169837) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1306818472105319) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.036450532329434085) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.016633706176686372) q[3];
cx q[2],q[3];
rx(-0.1796807139718286) q[0];
rz(-0.0007879619525901007) q[0];
rx(0.052517818349146934) q[1];
rz(-0.07646183926816028) q[1];
rx(-0.039677033393238005) q[2];
rz(-0.08858546384870687) q[2];
rx(-0.16891611958660618) q[3];
rz(-0.004229225741870958) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.14739384248351442) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1456135467810259) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.006202126374918904) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.15641544241686142) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04841681202810248) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.007276751029746284) q[3];
cx q[2],q[3];
rx(-0.1835588788424347) q[0];
rz(-0.011676781564992675) q[0];
rx(-0.01719927086727422) q[1];
rz(-0.1215313862622816) q[1];
rx(0.024728216307495592) q[2];
rz(-0.035996120763827605) q[2];
rx(-0.15040570540329865) q[3];
rz(-0.031415576891580387) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11769205290778557) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1454833518528898) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.04256029525623237) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.18489058971473485) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.15247211594619622) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.014561233316837192) q[3];
cx q[2],q[3];
rx(-0.17848800767845613) q[0];
rz(0.013821153576422896) q[0];
rx(0.010718366442647093) q[1];
rz(-0.12446433911657019) q[1];
rx(-0.03345414566258593) q[2];
rz(-0.09542784024315963) q[2];
rx(-0.18027812108980434) q[3];
rz(-0.01502560324787098) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.18685211546705713) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11583848250127085) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.08536948258650046) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11441306606942764) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08973462750426078) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.07490390807580491) q[3];
cx q[2],q[3];
rx(-0.17023082305060125) q[0];
rz(-0.0068047219285373986) q[0];
rx(0.04090270045016045) q[1];
rz(-0.06293486424824857) q[1];
rx(-0.07035468595563965) q[2];
rz(-0.07128556319248938) q[2];
rx(-0.08314784864079942) q[3];
rz(0.019368350318602533) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10084684770663528) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1283371330737034) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.06664403184647302) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09681458266723603) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08145015904639326) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.08089424655989025) q[3];
cx q[2],q[3];
rx(-0.15212426706955323) q[0];
rz(-0.012534313267513355) q[0];
rx(-0.03798730028995311) q[1];
rz(-0.08123439950054906) q[1];
rx(-0.02795722776757921) q[2];
rz(-0.04596968503772353) q[2];
rx(-0.04634336612183188) q[3];
rz(-0.0535745567933311) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09072316683985353) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.02760140836963983) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.1458184948164174) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.14002789549753686) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08387317980179011) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.02985733918983179) q[3];
cx q[2],q[3];
rx(-0.21448112917204015) q[0];
rz(0.017541190020147594) q[0];
rx(-0.002431089335838842) q[1];
rz(-0.021175840537763658) q[1];
rx(-0.07924410191038167) q[2];
rz(-0.08512679167313583) q[2];
rx(-0.09812510540223242) q[3];
rz(-0.03547952093378602) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0740853078329127) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07882320099924925) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.052757730173178086) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.13953013480510604) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09671121539850652) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.005777658053586043) q[3];
cx q[2],q[3];
rx(-0.20025243358356418) q[0];
rz(-0.012333559806250075) q[0];
rx(0.001000464263798881) q[1];
rz(-0.0473263619014772) q[1];
rx(-0.10476180781096762) q[2];
rz(-0.07384904218483464) q[2];
rx(-0.05084261638732974) q[3];
rz(-0.059185246619886535) q[3];