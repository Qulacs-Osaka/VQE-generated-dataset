OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.2750506217216678) q[0];
rz(-0.4097950440401804) q[0];
ry(1.7241228092329308) q[1];
rz(1.3395918995039828) q[1];
ry(-2.4931596813359893) q[2];
rz(2.7951755954952797) q[2];
ry(1.3177786564518987) q[3];
rz(1.0710299115515243) q[3];
ry(-0.0006814903839185149) q[4];
rz(2.4328383767139004) q[4];
ry(1.2674516704293628) q[5];
rz(0.7985558295097532) q[5];
ry(0.17420216097690935) q[6];
rz(-0.4789274327329226) q[6];
ry(-0.005460945257342857) q[7];
rz(0.7827016104812268) q[7];
ry(-1.567227387846831) q[8];
rz(0.7100612385040659) q[8];
ry(2.6625411515539716) q[9];
rz(-1.7754125546637527) q[9];
ry(-1.0807478890947824) q[10];
rz(2.9368429107339704) q[10];
ry(2.885150988838334) q[11];
rz(0.9525895438046026) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.505481974349514) q[0];
rz(0.3595573376267834) q[0];
ry(0.17292769399531083) q[1];
rz(0.24829243891946803) q[1];
ry(-0.2552844794103213) q[2];
rz(0.46413577867295325) q[2];
ry(-0.04804346189390429) q[3];
rz(-3.0384889980104375) q[3];
ry(3.107600546279905) q[4];
rz(-0.9587806382730903) q[4];
ry(-2.864992588558951) q[5];
rz(-1.383940297190091) q[5];
ry(-3.029845083976315) q[6];
rz(2.706483280159603) q[6];
ry(3.1412295103071974) q[7];
rz(0.5569903400467915) q[7];
ry(-2.290520206473317) q[8];
rz(-3.0727424816652125) q[8];
ry(-3.1402701267121844) q[9];
rz(0.7156185845523905) q[9];
ry(-3.113141853499576) q[10];
rz(2.94725729855463) q[10];
ry(0.0033373089283251466) q[11];
rz(-2.145762311692068) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.04637345140816863) q[0];
rz(-3.09547903514637) q[0];
ry(-2.8595044039631246) q[1];
rz(2.334318288290257) q[1];
ry(-1.7826803758105516) q[2];
rz(1.0402754624885717) q[2];
ry(3.002998028234791) q[3];
rz(0.42314627774759744) q[3];
ry(0.002015466715997505) q[4];
rz(-0.0990885425845125) q[4];
ry(2.626195455681388) q[5];
rz(0.5616760390965536) q[5];
ry(0.4179320984664532) q[6];
rz(-0.0648043564518568) q[6];
ry(-3.1381731889072233) q[7];
rz(-0.29426022877361496) q[7];
ry(-0.942148366944903) q[8];
rz(2.234975666477059) q[8];
ry(3.070749523523276) q[9];
rz(2.5379172024052807) q[9];
ry(2.0555035911378234) q[10];
rz(-2.165057231791619) q[10];
ry(3.096923653324513) q[11];
rz(-0.8511640663930744) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.9298166233679654) q[0];
rz(3.0399242658292316) q[0];
ry(0.4500919998988836) q[1];
rz(2.7775632136653265) q[1];
ry(-3.118536075351653) q[2];
rz(-2.310329464220839) q[2];
ry(-3.1052525059768383) q[3];
rz(1.8465372106903137) q[3];
ry(-2.6820917851633057) q[4];
rz(-0.9451294227974714) q[4];
ry(1.4705712519755267) q[5];
rz(2.1254226547109623) q[5];
ry(-2.990349349729831) q[6];
rz(-0.7992435058691175) q[6];
ry(0.0011184002007829422) q[7];
rz(1.7130806084138894) q[7];
ry(1.4750511076936466) q[8];
rz(1.4203628995281514) q[8];
ry(-0.3078546638091647) q[9];
rz(1.135875928578728) q[9];
ry(2.2699778612474546) q[10];
rz(2.1807811365657788) q[10];
ry(0.11059340061992538) q[11];
rz(0.7624042968191491) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.4319322032071094) q[0];
rz(1.2971630901776203) q[0];
ry(-1.222248066302706) q[1];
rz(2.6944076073948713) q[1];
ry(-0.7441316116130907) q[2];
rz(-1.238480435736438) q[2];
ry(-3.136483089936883) q[3];
rz(2.6845660935523137) q[3];
ry(-3.138856131372095) q[4];
rz(2.800327288341363) q[4];
ry(-0.11947836560122624) q[5];
rz(2.292113417349533) q[5];
ry(2.641672788225124) q[6];
rz(-1.595934100634972) q[6];
ry(-1.0482671782027797) q[7];
rz(1.9820163774238457) q[7];
ry(-1.574298062885605) q[8];
rz(2.547818574672328) q[8];
ry(2.8571195830859186) q[9];
rz(2.950864889471288) q[9];
ry(0.026824025458158648) q[10];
rz(-2.2530093999452054) q[10];
ry(3.1303462497783054) q[11];
rz(1.3529877610586754) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.6781938569189494) q[0];
rz(-1.7255520131007955) q[0];
ry(-2.4361262409836635) q[1];
rz(-2.5461770311572223) q[1];
ry(-1.5937347126394066) q[2];
rz(1.6194204134947496) q[2];
ry(2.9369914117104585) q[3];
rz(2.9601256563777683) q[3];
ry(-1.7834212755205339) q[4];
rz(3.100033986182118) q[4];
ry(2.583633200193999) q[5];
rz(-2.5417643176917224) q[5];
ry(0.9798656478553177) q[6];
rz(0.030200779976332078) q[6];
ry(2.8691264166729247) q[7];
rz(-0.0240109851003703) q[7];
ry(-3.140903694883554) q[8];
rz(2.411586680559214) q[8];
ry(-1.314121979992789) q[9];
rz(0.2121127541962901) q[9];
ry(-1.7799107939037802) q[10];
rz(-2.1221942235141182) q[10];
ry(-1.537968228991616) q[11];
rz(1.8801530861953548) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.4980099791159559) q[0];
rz(-2.664413861636658) q[0];
ry(0.12847235503233367) q[1];
rz(2.4641314046479135) q[1];
ry(3.1082546862785314) q[2];
rz(-1.5166444855038765) q[2];
ry(-3.1409822263141343) q[3];
rz(-2.0050021841886174) q[3];
ry(3.1370473628243554) q[4];
rz(-0.04357273834823902) q[4];
ry(-3.1105146142715356) q[5];
rz(-2.9077541280403874) q[5];
ry(-0.6741339641208496) q[6];
rz(3.0456357682167003) q[6];
ry(3.076852588721757) q[7];
rz(2.6714480095556756) q[7];
ry(-3.128933450342095) q[8];
rz(0.7382559275839301) q[8];
ry(0.42670672883317007) q[9];
rz(-3.140315169057796) q[9];
ry(0.007041583268520668) q[10];
rz(2.338186523441631) q[10];
ry(-0.18361288156014766) q[11];
rz(-1.6498523541838157) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.6695569502173983) q[0];
rz(-1.5550372344433825) q[0];
ry(-1.1703810961348524) q[1];
rz(-2.901578531702381) q[1];
ry(-1.9599436193654158) q[2];
rz(2.336428931503264) q[2];
ry(-1.2410052051100027) q[3];
rz(1.9053116888642112) q[3];
ry(2.5439049404803384) q[4];
rz(2.729379787795386) q[4];
ry(2.9905714282623355) q[5];
rz(-0.9400135835563335) q[5];
ry(1.0202176772830658) q[6];
rz(-2.137027232270125) q[6];
ry(-0.05236469689083555) q[7];
rz(-1.440608663917524) q[7];
ry(-0.006431963826311282) q[8];
rz(-1.715760405166423) q[8];
ry(-1.4286174892523613) q[9];
rz(-1.671165763226428) q[9];
ry(-0.8344162514624758) q[10];
rz(1.4695524044792927) q[10];
ry(-1.142911330882103) q[11];
rz(-0.7843173106884603) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.8998767116334158) q[0];
rz(1.1055320408912115) q[0];
ry(-0.7448394812900281) q[1];
rz(0.3989144941112953) q[1];
ry(-2.230465687086671) q[2];
rz(0.43924733440447616) q[2];
ry(-0.0015766005319096624) q[3];
rz(2.393492052387041) q[3];
ry(-3.1412279348953502) q[4];
rz(-1.8482127739251135) q[4];
ry(-2.7421097888438215) q[5];
rz(-2.352407608032439) q[5];
ry(2.578648400688172) q[6];
rz(2.3723740661833084) q[6];
ry(1.2721885074422152) q[7];
rz(2.2270691390269826) q[7];
ry(-1.586844153779552) q[8];
rz(0.2792753751638758) q[8];
ry(1.1361654094113414) q[9];
rz(-1.6725246199937365) q[9];
ry(-2.8597990102457014) q[10];
rz(-1.1151862088712938) q[10];
ry(1.990000969465439) q[11];
rz(2.370391583368609) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.2894152468988453) q[0];
rz(-2.433089170870915) q[0];
ry(0.20516892390777464) q[1];
rz(-0.8478050952084419) q[1];
ry(-0.19527547204333118) q[2];
rz(-0.22917886972337242) q[2];
ry(-1.887810291192367) q[3];
rz(-1.9499817811675115) q[3];
ry(-1.7858267956134866) q[4];
rz(-2.3441137703315253) q[4];
ry(0.23495304350773177) q[5];
rz(-0.4983931336584334) q[5];
ry(0.91839874430421) q[6];
rz(1.9787969978162305) q[6];
ry(0.6508686064521205) q[7];
rz(-0.4499160287362167) q[7];
ry(-0.003657192919370894) q[8];
rz(-2.765432120377111) q[8];
ry(-3.140907747605334) q[9];
rz(-1.7456583689646514) q[9];
ry(0.005163563374044777) q[10];
rz(0.8114807813908302) q[10];
ry(2.404057898027377) q[11];
rz(1.5608994172064938) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.012258092055468) q[0];
rz(-0.07424438897905716) q[0];
ry(-2.1317643560735773) q[1];
rz(0.9370668560457048) q[1];
ry(1.1491666585380882) q[2];
rz(-1.540239964042268) q[2];
ry(0.0008344261156993936) q[3];
rz(0.2714245771839687) q[3];
ry(0.01964918156845652) q[4];
rz(3.0373501536664715) q[4];
ry(0.7402579002837993) q[5];
rz(2.215950483736533) q[5];
ry(0.3783965700593264) q[6];
rz(-2.233727193889781) q[6];
ry(-2.64213277656413) q[7];
rz(-2.17069436189691) q[7];
ry(-2.53428664538877) q[8];
rz(-0.3980498550767107) q[8];
ry(2.0649853752377068) q[9];
rz(-2.044753833204743) q[9];
ry(2.7872853674974545) q[10];
rz(-2.9402899938133022) q[10];
ry(0.7916658206877498) q[11];
rz(-0.3941653030256589) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.749760004642758) q[0];
rz(-2.866745765295199) q[0];
ry(0.6669118029818506) q[1];
rz(-2.4973734423046468) q[1];
ry(-2.361786766690381) q[2];
rz(0.7745042443929231) q[2];
ry(3.0521572828860744) q[3];
rz(1.0340987228745668) q[3];
ry(1.8558747188923934) q[4];
rz(-2.863836425284115) q[4];
ry(0.004188746985603049) q[5];
rz(0.7185414093885569) q[5];
ry(-2.9999492402582306) q[6];
rz(-0.5483000850217977) q[6];
ry(1.200115630432442) q[7];
rz(0.27944577337312104) q[7];
ry(1.164854692617196) q[8];
rz(-2.2143455063575743) q[8];
ry(0.008487935212158071) q[9];
rz(-3.056521751897631) q[9];
ry(-1.4876961404046558) q[10];
rz(0.23326317904266247) q[10];
ry(-2.7032439438018807) q[11];
rz(-1.4060292472009843) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.889464296412532) q[0];
rz(2.810662830425421) q[0];
ry(-1.1806096144876117) q[1];
rz(-1.6224862267310538) q[1];
ry(2.757846855858201) q[2];
rz(-2.1239051352668206) q[2];
ry(-0.0025711355000417615) q[3];
rz(-1.1518427637434847) q[3];
ry(0.022887341673142103) q[4];
rz(1.2988447032524393) q[4];
ry(-1.7640773943380523) q[5];
rz(0.018065610065606065) q[5];
ry(-1.210940079280482) q[6];
rz(0.38024036296112357) q[6];
ry(1.6631161798820342) q[7];
rz(-2.3653483805488023) q[7];
ry(3.0944313820079206) q[8];
rz(-0.4991102060107876) q[8];
ry(3.13944229688048) q[9];
rz(-2.23729012427519) q[9];
ry(-0.8312723286468605) q[10];
rz(-0.09670607996108531) q[10];
ry(-0.07789545958662154) q[11];
rz(1.9794262847868653) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.6401163950905358) q[0];
rz(-0.6926949968577264) q[0];
ry(-1.2325997099134929) q[1];
rz(-1.5210288993099723) q[1];
ry(0.14500104782938641) q[2];
rz(-0.6912847882545196) q[2];
ry(-1.1231276660861313) q[3];
rz(1.646869875299009) q[3];
ry(-2.500807889709105) q[4];
rz(1.4519573663376981) q[4];
ry(-3.1393562851149324) q[5];
rz(0.658772372318939) q[5];
ry(1.5027110334855294) q[6];
rz(-2.9077313141269805) q[6];
ry(-2.962615058003255) q[7];
rz(-2.637998180291229) q[7];
ry(-0.5837485902921347) q[8];
rz(-0.976700964458419) q[8];
ry(0.0025666534539201535) q[9];
rz(-0.8089811388786236) q[9];
ry(1.609252780012782) q[10];
rz(-2.082247085535262) q[10];
ry(-3.0697505810050423) q[11];
rz(0.6598408926793072) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.8274764983480507) q[0];
rz(-1.1252310721099066) q[0];
ry(1.4778516892326747) q[1];
rz(-2.16870542120009) q[1];
ry(0.03138500522771359) q[2];
rz(2.481828016788716) q[2];
ry(-3.1217562677314072) q[3];
rz(1.277160449506811) q[3];
ry(-3.137218059125586) q[4];
rz(-1.8495158705522357) q[4];
ry(3.135381265582254) q[5];
rz(-1.5329262356592128) q[5];
ry(-3.0310938983883218) q[6];
rz(-3.081679565348214) q[6];
ry(0.01701337494989324) q[7];
rz(-2.109607218313131) q[7];
ry(-2.9475194415244004) q[8];
rz(-2.882357224305458) q[8];
ry(-3.088001637668637) q[9];
rz(0.33652924635825276) q[9];
ry(2.381904812275059) q[10];
rz(-2.257236138444913) q[10];
ry(1.427235089660054) q[11];
rz(-0.39059054892465195) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.2235207931628131) q[0];
rz(2.922243302740427) q[0];
ry(1.707664175662888) q[1];
rz(2.262589001351229) q[1];
ry(-3.119139817650372) q[2];
rz(0.8974553526469798) q[2];
ry(2.3190017021854015) q[3];
rz(-2.6147582662210023) q[3];
ry(0.415733040949984) q[4];
rz(-2.5283910392660243) q[4];
ry(-3.1407281005291177) q[5];
rz(-1.6453767427543982) q[5];
ry(1.252923765073308) q[6];
rz(-2.584398346591761) q[6];
ry(-0.5479970749732317) q[7];
rz(2.623071591626405) q[7];
ry(-1.7822097151568292) q[8];
rz(-0.23342014297629735) q[8];
ry(-3.1389290940782413) q[9];
rz(2.6748535767754897) q[9];
ry(-3.135708248306888) q[10];
rz(-2.8290467279621643) q[10];
ry(2.9235402725675885) q[11];
rz(-0.6648750218032715) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.4576748827481443) q[0];
rz(0.41916827677948726) q[0];
ry(-2.999567404383319) q[1];
rz(1.028504045571826) q[1];
ry(-2.567837700933729) q[2];
rz(2.047555515132709) q[2];
ry(0.37866559695778734) q[3];
rz(-2.237819308631263) q[3];
ry(0.011931729122581558) q[4];
rz(-1.4669283470208025) q[4];
ry(-1.2317463789417404) q[5];
rz(-2.284235754160524) q[5];
ry(0.2727672689609548) q[6];
rz(3.051297631520773) q[6];
ry(-3.13346424199826) q[7];
rz(2.277055675995684) q[7];
ry(0.1919319719537782) q[8];
rz(-2.7042793316321556) q[8];
ry(0.9501551862613519) q[9];
rz(-2.9061791329921327) q[9];
ry(2.91160071061877) q[10];
rz(1.5158940997062125) q[10];
ry(-1.899776824189284) q[11];
rz(-0.2500614911381378) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.18057218023565) q[0];
rz(1.051088869701724) q[0];
ry(-2.2138091514584826) q[1];
rz(-3.046236412348173) q[1];
ry(3.1215615530525875) q[2];
rz(-1.0643322411047251) q[2];
ry(0.04224116635570209) q[3];
rz(2.345175541853763) q[3];
ry(-3.141065348451177) q[4];
rz(0.5017693755517559) q[4];
ry(0.0015579133660308386) q[5];
rz(-0.864942693635587) q[5];
ry(0.0753392720575332) q[6];
rz(0.11124636728307634) q[6];
ry(-1.9443135489643053) q[7];
rz(-2.073293178974743) q[7];
ry(0.02213728390356423) q[8];
rz(2.8444077131169263) q[8];
ry(0.001527357606134164) q[9];
rz(-0.22045683631052643) q[9];
ry(3.140896347473652) q[10];
rz(1.2361307030464914) q[10];
ry(3.0902775167180425) q[11];
rz(1.0590978920934377) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.9394917076159064) q[0];
rz(-1.5200606438479183) q[0];
ry(0.02640041582364076) q[1];
rz(0.9279437381510061) q[1];
ry(2.274824163123733) q[2];
rz(-2.717551855513627) q[2];
ry(-2.797392378876785) q[3];
rz(0.45224692456877236) q[3];
ry(0.017952170212347504) q[4];
rz(-2.791883948858813) q[4];
ry(0.9955436900532905) q[5];
rz(0.2823956369063661) q[5];
ry(1.2330836111203283) q[6];
rz(0.10333146822436301) q[6];
ry(0.052411716381310924) q[7];
rz(1.8111056368597331) q[7];
ry(-0.8349970029670574) q[8];
rz(1.9897435684102116) q[8];
ry(2.208498263944411) q[9];
rz(-0.441903195449364) q[9];
ry(1.2984746436983894) q[10];
rz(-2.2409413011091313) q[10];
ry(2.062648254639522) q[11];
rz(-2.691696712273782) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.3994177612055263) q[0];
rz(-2.2273126581873885) q[0];
ry(-0.011198935227057838) q[1];
rz(1.6069585343269885) q[1];
ry(0.8800077730588365) q[2];
rz(1.8636807520417789) q[2];
ry(0.8258527923298794) q[3];
rz(1.711921041890186) q[3];
ry(-0.00963616785341248) q[4];
rz(-2.8307036819081843) q[4];
ry(-3.1365022937460196) q[5];
rz(-1.9744502354046953) q[5];
ry(-0.3488918537822592) q[6];
rz(0.087427614720748) q[6];
ry(1.7517945817563039) q[7];
rz(2.979173864693128) q[7];
ry(1.210649754381433) q[8];
rz(-2.0824423125074825) q[8];
ry(-0.0011751046778857497) q[9];
rz(-2.40484913407498) q[9];
ry(-0.020986353417660197) q[10];
rz(-0.12184510817501558) q[10];
ry(-1.3694208274118458) q[11];
rz(0.7990642816521235) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.792370456359871) q[0];
rz(1.641088182617665) q[0];
ry(-2.757312558757213) q[1];
rz(1.5621337244576825) q[1];
ry(0.04154941909127085) q[2];
rz(-1.6131609500370168) q[2];
ry(-0.7274145311186571) q[3];
rz(-2.228113335223968) q[3];
ry(1.702736469138344) q[4];
rz(-2.925726011630703) q[4];
ry(-2.3757622341231075) q[5];
rz(-1.2066211403986684) q[5];
ry(-2.699501828025289) q[6];
rz(-1.6434784121571016) q[6];
ry(2.5094880115233678) q[7];
rz(1.799318414466524) q[7];
ry(1.9428938264667863) q[8];
rz(1.5239335441052495) q[8];
ry(0.13554455813883504) q[9];
rz(-2.099014394220349) q[9];
ry(-3.1401085234334687) q[10];
rz(1.594910619334051) q[10];
ry(2.031841617564339) q[11];
rz(-0.1931888367878753) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.2737574281666943) q[0];
rz(0.5906372392804771) q[0];
ry(0.051171759464158424) q[1];
rz(-1.750796149663778) q[1];
ry(3.1045241451124164) q[2];
rz(-1.514784429764438) q[2];
ry(-3.095984979274037) q[3];
rz(2.7942313194017245) q[3];
ry(-3.0757743291904056) q[4];
rz(-2.9328480699797477) q[4];
ry(3.136070709519399) q[5];
rz(-0.12256590679738456) q[5];
ry(-3.1303132278111043) q[6];
rz(-1.9118402927068567) q[6];
ry(2.4459189368973635) q[7];
rz(2.727116376904803) q[7];
ry(-2.7789706860662413) q[8];
rz(-2.463938914406518) q[8];
ry(0.3031670301218227) q[9];
rz(-1.2324330210943) q[9];
ry(0.01624176175974057) q[10];
rz(-0.4597800713600826) q[10];
ry(-2.9960078666730636) q[11];
rz(0.7920212269556428) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.8074915389603823) q[0];
rz(-2.762463316000216) q[0];
ry(-3.083874973618138) q[1];
rz(2.842630757985884) q[1];
ry(0.05539320980344976) q[2];
rz(-1.2185402252358755) q[2];
ry(2.1833099059612584) q[3];
rz(-2.671264340754774) q[3];
ry(1.0138427389748825) q[4];
rz(2.9571309939914614) q[4];
ry(3.0172316840105795) q[5];
rz(2.173148919572868) q[5];
ry(2.8470936928935346) q[6];
rz(1.0034358868349305) q[6];
ry(1.6768090300852814) q[7];
rz(1.2203215282551814) q[7];
ry(-0.027089919411536587) q[8];
rz(1.2296242251662122) q[8];
ry(0.35486524443000317) q[9];
rz(1.8162238711750671) q[9];
ry(-1.467770999090573) q[10];
rz(1.7360532340907024) q[10];
ry(-2.1848895865554256) q[11];
rz(0.9594120831050804) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.043358675052504) q[0];
rz(-2.545929479518497) q[0];
ry(0.7366511658028684) q[1];
rz(-1.8702277434271002) q[1];
ry(-0.009855328923779183) q[2];
rz(-0.7361864241233342) q[2];
ry(0.1896835430281376) q[3];
rz(0.10635378232444381) q[3];
ry(0.10865103263419895) q[4];
rz(2.1813450377709067) q[4];
ry(3.139411449068539) q[5];
rz(0.36723632872226064) q[5];
ry(-3.1099247652544855) q[6];
rz(1.532694098440188) q[6];
ry(-0.5867689340760025) q[7];
rz(0.2677050283810921) q[7];
ry(-0.0662846954813281) q[8];
rz(-0.8519267120619859) q[8];
ry(-0.03435810498802051) q[9];
rz(-2.4622902771407738) q[9];
ry(-0.0013857556503652075) q[10];
rz(1.6147287992216581) q[10];
ry(2.2844028705640587) q[11];
rz(0.5750302047130766) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.19298968516248) q[0];
rz(1.5051237289287025) q[0];
ry(0.021168358563154577) q[1];
rz(-1.674110389007979) q[1];
ry(3.048619686069663) q[2];
rz(-0.2326097819985451) q[2];
ry(2.6754821295084668) q[3];
rz(-1.0417225461338708) q[3];
ry(3.118015715796292) q[4];
rz(2.152661676561646) q[4];
ry(-3.139442837922215) q[5];
rz(3.0127117503913636) q[5];
ry(-0.7051384396445568) q[6];
rz(2.556020716895415) q[6];
ry(1.418179356853688) q[7];
rz(-2.367989340967428) q[7];
ry(-0.043235638928528) q[8];
rz(-2.4820517006180256) q[8];
ry(-1.7725735740146424) q[9];
rz(2.1337314572986403) q[9];
ry(-0.08846174894424763) q[10];
rz(-3.0265101171996664) q[10];
ry(-2.87153305075383) q[11];
rz(-2.208471442027421) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.9789394686721684) q[0];
rz(2.773093716977861) q[0];
ry(-2.852854206079257) q[1];
rz(2.001985795302212) q[1];
ry(0.14178058906245106) q[2];
rz(-0.29549052378053386) q[2];
ry(0.07080301307114033) q[3];
rz(-2.4228509257438224) q[3];
ry(2.9673761344030996) q[4];
rz(0.35277663586252406) q[4];
ry(-0.0028964665059518784) q[5];
rz(1.8294923960911464) q[5];
ry(-3.0592640816050625) q[6];
rz(2.3362786924149224) q[6];
ry(-1.2131569489708798) q[7];
rz(-0.9087610846238299) q[7];
ry(0.3506336091669415) q[8];
rz(2.974231098608389) q[8];
ry(-0.023449060358535233) q[9];
rz(1.8557042660339729) q[9];
ry(-2.9540998774179634) q[10];
rz(1.425195698170021) q[10];
ry(-1.9124502078416508) q[11];
rz(-0.790065991184429) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.1025038300853742) q[0];
rz(2.4562512431370243) q[0];
ry(0.15540452493569304) q[1];
rz(-0.1633730927849265) q[1];
ry(-2.2756163620130976) q[2];
rz(3.0323593794571497) q[2];
ry(2.783133121257621) q[3];
rz(2.157434899130953) q[3];
ry(-0.026243324720450545) q[4];
rz(1.5933814702211526) q[4];
ry(0.6149228706988348) q[5];
rz(2.5896246024091205) q[5];
ry(3.0527091889779525) q[6];
rz(-0.5695763024401019) q[6];
ry(-0.002525822237923765) q[7];
rz(1.1463491053768915) q[7];
ry(3.140901859744765) q[8];
rz(2.812218968274062) q[8];
ry(-3.1412204774573516) q[9];
rz(-2.6579015675282442) q[9];
ry(-2.547286277491393) q[10];
rz(-1.899154998748717) q[10];
ry(-1.1088586773451263) q[11];
rz(-1.4715240064508954) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.6664537489938374) q[0];
rz(-2.259467135652811) q[0];
ry(-3.1383416587158126) q[1];
rz(-1.7065665025001584) q[1];
ry(0.0037705275113994437) q[2];
rz(-2.6209730952772348) q[2];
ry(-3.0094739008935822) q[3];
rz(-2.7780343245949584) q[3];
ry(-0.9758317998277574) q[4];
rz(1.7555530759341167) q[4];
ry(0.01807190289533003) q[5];
rz(-2.2432938926073707) q[5];
ry(-0.005759426182181043) q[6];
rz(2.844248762630652) q[6];
ry(1.4319024490904804) q[7];
rz(0.9431749385951793) q[7];
ry(0.36337558014952637) q[8];
rz(0.30796896543376373) q[8];
ry(0.023585931522135804) q[9];
rz(2.0197805035530014) q[9];
ry(-3.054509950515616) q[10];
rz(-2.8815083125904852) q[10];
ry(1.6514589444520373) q[11];
rz(-1.0776588144387915) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.42165979726512) q[0];
rz(-1.3429572859325272) q[0];
ry(-2.5612248767095607) q[1];
rz(-2.023248148774986) q[1];
ry(-1.2585256178051232) q[2];
rz(-2.5764889942179354) q[2];
ry(-0.00434089039238561) q[3];
rz(1.6399880619923606) q[3];
ry(-1.316231620745298) q[4];
rz(0.045847749002172485) q[4];
ry(-3.134521175442597) q[5];
rz(-1.1378119626903898) q[5];
ry(2.987208452826847) q[6];
rz(1.5617444349385954) q[6];
ry(0.6198383319503318) q[7];
rz(0.988149920695835) q[7];
ry(-2.0294992175225106) q[8];
rz(-2.2012632681073176) q[8];
ry(1.654841583867517) q[9];
rz(-1.749611198997795) q[9];
ry(0.9576750847071859) q[10];
rz(-2.122500551222789) q[10];
ry(0.5768386361488638) q[11];
rz(2.9179779417939398) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.2907060663555334) q[0];
rz(-2.2788081237555264) q[0];
ry(-2.986548973937958) q[1];
rz(2.8353304039980363) q[1];
ry(3.128660651874011) q[2];
rz(-2.635753969636293) q[2];
ry(-3.1377309954912227) q[3];
rz(-0.23452561526055146) q[3];
ry(-1.8042627396751145) q[4];
rz(0.008578412342360231) q[4];
ry(3.1369085275462982) q[5];
rz(-2.4894630729113185) q[5];
ry(-2.935003242695651) q[6];
rz(2.665422346462358) q[6];
ry(-3.1342610022178694) q[7];
rz(2.0710403129491777) q[7];
ry(0.0015636443400976734) q[8];
rz(-2.679749194318963) q[8];
ry(-3.137729886541618) q[9];
rz(1.5945578952600912) q[9];
ry(-2.7331663066371896) q[10];
rz(1.4447547416718123) q[10];
ry(-0.936892356692421) q[11];
rz(-2.387217650042484) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.1302155809133172) q[0];
rz(1.1342108419499624) q[0];
ry(-0.29039223396663516) q[1];
rz(1.4247276982890493) q[1];
ry(0.1592977696938922) q[2];
rz(-0.9738742249275196) q[2];
ry(3.1097480707743093) q[3];
rz(1.4644209707250782) q[3];
ry(-1.8297161386811762) q[4];
rz(-1.028647231533656) q[4];
ry(0.0002619329639928092) q[5];
rz(1.7834783471041913) q[5];
ry(3.0040044970876876) q[6];
rz(-3.0622615410472958) q[6];
ry(-0.6253801610089852) q[7];
rz(-2.110875036046454) q[7];
ry(1.9078154603956374) q[8];
rz(-2.408333264439048) q[8];
ry(1.241431875960955) q[9];
rz(0.8371581352198206) q[9];
ry(3.128635103613984) q[10];
rz(-2.62810008975009) q[10];
ry(-1.3524092734986595) q[11];
rz(-0.810787954053089) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.7626799420929302) q[0];
rz(-1.9334822667803546) q[0];
ry(2.8860685420345025) q[1];
rz(0.4442698109877279) q[1];
ry(-3.139693422025499) q[2];
rz(-1.478575681866218) q[2];
ry(3.0241749943832774) q[3];
rz(0.7354936642504208) q[3];
ry(-1.8139661559989542) q[4];
rz(-2.823267122482076) q[4];
ry(-3.1199328523510887) q[5];
rz(2.2113807242955055) q[5];
ry(-0.1652197523556573) q[6];
rz(0.5769016486321414) q[6];
ry(3.107145557523169) q[7];
rz(-0.5424390422502132) q[7];
ry(-0.0022751565071388313) q[8];
rz(1.1486181465940923) q[8];
ry(-3.1257241026841966) q[9];
rz(0.6887611608803436) q[9];
ry(0.042994306165024296) q[10];
rz(1.246155195964972) q[10];
ry(1.638835708740801) q[11];
rz(3.0616391395238605) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.6556693294801876) q[0];
rz(2.9861325757487016) q[0];
ry(1.6227369707811057) q[1];
rz(2.5934666884358237) q[1];
ry(0.0013175021341393018) q[2];
rz(-1.6121378736270087) q[2];
ry(-0.41799126912631746) q[3];
rz(2.6807096597087003) q[3];
ry(2.846036887733669) q[4];
rz(-1.7418486125506867) q[4];
ry(0.05389862942763113) q[5];
rz(-3.0792936333965453) q[5];
ry(2.4726601045585888) q[6];
rz(0.5782172668776814) q[6];
ry(2.7618190329535763) q[7];
rz(1.236746933957668) q[7];
ry(-1.7373545997675153) q[8];
rz(1.2265150032122487) q[8];
ry(-0.889314765019308) q[9];
rz(-1.6559824283899165) q[9];
ry(2.899765754770126) q[10];
rz(2.259043674170109) q[10];
ry(-2.8952340470952476) q[11];
rz(2.7061830787636723) q[11];