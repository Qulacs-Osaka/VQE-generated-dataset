OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(3.135405981468049) q[0];
rz(-1.9548980227345139) q[0];
ry(-0.03566337256567654) q[1];
rz(-1.2378507182171) q[1];
ry(1.6121043139594553) q[2];
rz(1.329751633593366) q[2];
ry(-1.5392932192622777) q[3];
rz(1.4673371471577106) q[3];
ry(3.1127517417181156) q[4];
rz(0.891215000397362) q[4];
ry(0.02667757313532411) q[5];
rz(3.0939667110995317) q[5];
ry(0.14519360759590272) q[6];
rz(1.4982099248841392) q[6];
ry(-1.884505955672733) q[7];
rz(-1.5677641198776888) q[7];
ry(1.3990689596472323) q[8];
rz(-0.3686208767354815) q[8];
ry(-1.617101734288472) q[9];
rz(-2.813825218617013) q[9];
ry(-1.5719583132986985) q[10];
rz(3.060067762288518) q[10];
ry(-1.5685977630910024) q[11];
rz(1.028616048715296) q[11];
ry(-1.5690765913041458) q[12];
rz(-1.5522650620499274) q[12];
ry(-0.1715118034524874) q[13];
rz(0.04077864370072006) q[13];
ry(3.141588715263698) q[14];
rz(1.6151750404713052) q[14];
ry(-3.1415772591603557) q[15];
rz(-1.9077889422447836) q[15];
ry(1.5562192580394385) q[16];
rz(-1.7427174555804976) q[16];
ry(-1.5776006708215393) q[17];
rz(1.8323119773450445) q[17];
ry(-3.0955062336163697) q[18];
rz(-2.381513615183019) q[18];
ry(-3.0888395027098157) q[19];
rz(-2.1403531375651355) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.4845817283514213) q[0];
rz(1.2738217370033924) q[0];
ry(-0.8987610260813301) q[1];
rz(-1.5661773191841757) q[1];
ry(-1.4555433816976493) q[2];
rz(-0.9145248445232291) q[2];
ry(1.3377732506194366) q[3];
rz(-2.7864622990610326) q[3];
ry(3.0765136973378935) q[4];
rz(-2.9385349051965903) q[4];
ry(0.02789279251742224) q[5];
rz(-1.1716238081085475) q[5];
ry(1.5015119862880522) q[6];
rz(2.544887482640607) q[6];
ry(1.5636431799190396) q[7];
rz(0.0981073999517601) q[7];
ry(-3.1415474633513023) q[8];
rz(-1.9460074152118325) q[8];
ry(3.141455455111615) q[9];
rz(1.898269928554579) q[9];
ry(8.37511075504338e-05) q[10];
rz(1.650655733117025) q[10];
ry(-3.141535575612291) q[11];
rz(2.504876042049626) q[11];
ry(-3.116275188557091) q[12];
rz(-1.5545519353502066) q[12];
ry(1.5686206340594036) q[13];
rz(-1.9870728333771037) q[13];
ry(3.1012365160436315) q[14];
rz(-2.695058967944714) q[14];
ry(-0.01243985677366373) q[15];
rz(2.58688258550385) q[15];
ry(-1.5089140717389418) q[16];
rz(-0.31235680432344903) q[16];
ry(1.3787513848184876) q[17];
rz(-2.531555934864999) q[17];
ry(1.5519601795159175) q[18];
rz(-0.054012155413468044) q[18];
ry(1.6019272959045492) q[19];
rz(-2.9901759141348325) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.434676179127722) q[0];
rz(2.4412138495810156) q[0];
ry(1.8616378769938826) q[1];
rz(1.5088806733661473) q[1];
ry(-3.0972794745724403) q[2];
rz(-0.03267420329564796) q[2];
ry(-0.014273150243776735) q[3];
rz(-0.9015867412413971) q[3];
ry(3.132506869813214) q[4];
rz(-0.7232453276757921) q[4];
ry(-0.0032187891702406546) q[5];
rz(2.0734941096067496) q[5];
ry(1.7232892524779677) q[6];
rz(0.5906443556411595) q[6];
ry(0.9660112523800857) q[7];
rz(-1.753595422055546) q[7];
ry(-2.8871999062861318) q[8];
rz(-3.040762642474327) q[8];
ry(1.8281161014099858) q[9];
rz(-2.650751299986208) q[9];
ry(2.3812468988037185) q[10];
rz(1.5758525631110403) q[10];
ry(-2.3753334814061624) q[11];
rz(-1.9459439889001169) q[11];
ry(-1.586020330614497) q[12];
rz(-1.574375540018864) q[12];
ry(3.1391965350008073) q[13];
rz(2.7820934395189356) q[13];
ry(1.5928988009463287) q[14];
rz(-1.5556006245996947) q[14];
ry(-1.5856592250627202) q[15];
rz(-0.19045732478546434) q[15];
ry(-0.3274209275343506) q[16];
rz(-1.0550239789353055) q[16];
ry(-2.947661179007495) q[17];
rz(-2.655423999970984) q[17];
ry(0.39841133667501616) q[18];
rz(-0.5893465428525002) q[18];
ry(-2.7424821851307817) q[19];
rz(2.582125622306938) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.2448044919176313) q[0];
rz(1.1996962396787534) q[0];
ry(0.5393433475707372) q[1];
rz(2.2158012615468348) q[1];
ry(0.969619708734605) q[2];
rz(1.9390386027815159) q[2];
ry(1.0551534762381891) q[3];
rz(0.7184443573498163) q[3];
ry(1.4107164448235432) q[4];
rz(0.14597261105577164) q[4];
ry(-1.3966512800710085) q[5];
rz(3.0572228346805415) q[5];
ry(-3.131436735185747) q[6];
rz(1.7644610113086527) q[6];
ry(-1.560014239275813) q[7];
rz(-1.3746154949162808) q[7];
ry(-3.0951266079023205) q[8];
rz(0.30801007189468743) q[8];
ry(3.1413868569965904) q[9];
rz(0.5314317672099973) q[9];
ry(-2.365645367383191) q[10];
rz(2.3902154143856924) q[10];
ry(2.340988700246573) q[11];
rz(-1.8456287407064007) q[11];
ry(1.570062370910816) q[12];
rz(-0.007479780392510552) q[12];
ry(-1.5703014463859326) q[13];
rz(0.018500155445644314) q[13];
ry(-0.1864267619868505) q[14];
rz(-2.3404946430048637) q[14];
ry(-2.927302500778309) q[15];
rz(-0.11866723172576554) q[15];
ry(-1.5632103664488348) q[16];
rz(3.1387813678568235) q[16];
ry(1.570041741526626) q[17];
rz(-3.032249603623904) q[17];
ry(-3.117991750731892) q[18];
rz(-2.7856234657834382) q[18];
ry(0.022311110647660115) q[19];
rz(-0.050627657891070044) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.493373986016139) q[0];
rz(0.8584725385477362) q[0];
ry(1.5895694107581653) q[1];
rz(0.5655765368806992) q[1];
ry(0.15164437130244224) q[2];
rz(-1.5475982529732528) q[2];
ry(3.1165416945452793) q[3];
rz(-2.755660737727578) q[3];
ry(0.41519296790391014) q[4];
rz(1.3490105693023815) q[4];
ry(1.9753220285369215) q[5];
rz(-0.1790788349675445) q[5];
ry(-2.3288750603536883) q[6];
rz(-2.656524810591683) q[6];
ry(-0.8500275935310526) q[7];
rz(0.44382556042301835) q[7];
ry(-2.294771658981633) q[8];
rz(-2.6471612276483683) q[8];
ry(-2.432061740893776) q[9];
rz(-1.3408738681329853) q[9];
ry(-2.946265329358328) q[10];
rz(-1.9455021771533005) q[10];
ry(-0.00424372434554865) q[11];
rz(0.267028853278273) q[11];
ry(1.6703894111067763) q[12];
rz(2.843680244955405) q[12];
ry(-1.6681083575935993) q[13];
rz(-1.3100534877916905) q[13];
ry(-3.096624481727193) q[14];
rz(-1.214728114334184) q[14];
ry(-3.1054374913117604) q[15];
rz(0.9515594272959161) q[15];
ry(-1.202082868156637) q[16];
rz(2.746739085637474) q[16];
ry(1.8800458447751138) q[17];
rz(1.9696905487555343) q[17];
ry(-0.7456422796970339) q[18];
rz(0.8666199146501989) q[18];
ry(-1.2151493718659165) q[19];
rz(-1.1366556873233584) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.9016985240030233) q[0];
rz(-1.3484439560390742) q[0];
ry(-3.0630845495279146) q[1];
rz(-0.5469885429651518) q[1];
ry(-0.818157875972463) q[2];
rz(-1.5761799942000572) q[2];
ry(2.3376334604217797) q[3];
rz(-1.2655851710035235) q[3];
ry(1.5149301332944474) q[4];
rz(-1.696573540630643) q[4];
ry(-2.876718895971567) q[5];
rz(2.7381721210316385) q[5];
ry(1.1914719745912694) q[6];
rz(0.6000199765353376) q[6];
ry(-1.9641117064382414) q[7];
rz(2.5485053066269794) q[7];
ry(-3.1396980411369952) q[8];
rz(-2.870815745107216) q[8];
ry(0.002062556661934778) q[9];
rz(1.37484326402706) q[9];
ry(-0.035129285341820804) q[10];
rz(-1.9381370597503191) q[10];
ry(-2.9253216566081077) q[11];
rz(-1.535958077510584) q[11];
ry(-0.060691095532719966) q[12];
rz(0.1369937263464518) q[12];
ry(3.137773028799959) q[13];
rz(0.8994178457350078) q[13];
ry(2.235663926982207) q[14];
rz(-2.5257778616702677) q[14];
ry(-2.4681607121273603) q[15];
rz(0.9193933946416335) q[15];
ry(-0.841595859840151) q[16];
rz(-1.8850429373006967) q[16];
ry(-2.699475591175991) q[17];
rz(1.0260611683536816) q[17];
ry(2.0077381331137745) q[18];
rz(-0.5289282632090905) q[18];
ry(1.9852846930936672) q[19];
rz(2.589122944251648) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.356839381041748) q[0];
rz(1.1863850036308223) q[0];
ry(2.379072181049324) q[1];
rz(1.6869468985119989) q[1];
ry(3.1251688853936592) q[2];
rz(1.9903057955510686) q[2];
ry(-0.13223879403843608) q[3];
rz(-2.3501849819470118) q[3];
ry(-0.7098970819740884) q[4];
rz(2.575215424382902) q[4];
ry(0.9183783697894166) q[5];
rz(0.8899595302339431) q[5];
ry(-1.5907083593484777) q[6];
rz(-2.358945059487284) q[6];
ry(1.564658085276717) q[7];
rz(0.7692584662807977) q[7];
ry(-0.8370191852744419) q[8];
rz(-1.9985643022431434) q[8];
ry(0.6916543740395403) q[9];
rz(0.630432685643914) q[9];
ry(-1.5409906263536544) q[10];
rz(-0.8242146026352657) q[10];
ry(-1.5716679245764331) q[11];
rz(0.8424014515800902) q[11];
ry(-1.608973932547606) q[12];
rz(1.2071004177337015) q[12];
ry(3.1348629727074666) q[13];
rz(-0.04180844020494544) q[13];
ry(-0.004567932591212333) q[14];
rz(-0.8163906206316699) q[14];
ry(0.002908413967182888) q[15];
rz(-0.18431623303496011) q[15];
ry(2.4372662929068944) q[16];
rz(-0.9100502960115842) q[16];
ry(0.40744599438617135) q[17];
rz(2.1964021045095525) q[17];
ry(-1.5236416544339075) q[18];
rz(1.66117945929295) q[18];
ry(1.5142304639317778) q[19];
rz(1.6560177889893888) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.5062477634196412) q[0];
rz(-1.8966491192351178) q[0];
ry(2.6492093555082197) q[1];
rz(1.1178734657851117) q[1];
ry(-2.497130121305293) q[2];
rz(-2.0059132266943926) q[2];
ry(-0.4896460220126219) q[3];
rz(1.1891236647336925) q[3];
ry(-0.9885893719519504) q[4];
rz(-1.250478127021207) q[4];
ry(1.127895178797007) q[5];
rz(0.034447288316787304) q[5];
ry(-0.6105488616080165) q[6];
rz(-0.7996025744472282) q[6];
ry(0.5931962184053319) q[7];
rz(2.3574621704091863) q[7];
ry(-1.2582963914647642) q[8];
rz(-1.3987615603830728) q[8];
ry(-2.691665055481595) q[9];
rz(-2.4259140972925923) q[9];
ry(-2.6738601018299017) q[10];
rz(-2.1087095287352) q[10];
ry(-2.660848395521309) q[11];
rz(-2.1527678059344075) q[11];
ry(1.1669912920115006) q[12];
rz(1.772952317476384) q[12];
ry(2.600329784545152) q[13];
rz(1.0277921242071981) q[13];
ry(1.8390845528146766) q[14];
rz(-2.985197720520897) q[14];
ry(-1.8435747055469562) q[15];
rz(0.18571663491318163) q[15];
ry(1.628332002481961) q[16];
rz(0.7539922994332093) q[16];
ry(0.7324999727565457) q[17];
rz(0.015540149465665488) q[17];
ry(1.553531075041314) q[18];
rz(2.5928815508614655) q[18];
ry(-1.4987668544552646) q[19];
rz(-0.5110158356710245) q[19];