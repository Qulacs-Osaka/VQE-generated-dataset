OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-0.7875285014057577) q[0];
ry(0.6189896080480306) q[1];
cx q[0],q[1];
ry(0.9975169846255492) q[0];
ry(2.234088460395914) q[1];
cx q[0],q[1];
ry(2.878479188349696) q[2];
ry(0.9182217022382521) q[3];
cx q[2],q[3];
ry(0.4015025238582002) q[2];
ry(-0.36057083970246584) q[3];
cx q[2],q[3];
ry(-0.21400900920080088) q[4];
ry(-2.2145179191406026) q[5];
cx q[4],q[5];
ry(2.888235911554046) q[4];
ry(-1.507219471662907) q[5];
cx q[4],q[5];
ry(-0.1329264432344303) q[6];
ry(1.3503428325283313) q[7];
cx q[6],q[7];
ry(-3.012426810501131) q[6];
ry(-1.328349007985369) q[7];
cx q[6],q[7];
ry(0.2238741215483738) q[8];
ry(-0.03517775531965768) q[9];
cx q[8],q[9];
ry(-1.3150334443055325) q[8];
ry(1.6601316799136128) q[9];
cx q[8],q[9];
ry(-1.6129466286380314) q[10];
ry(-1.7606636798810058) q[11];
cx q[10],q[11];
ry(1.7823630835083524) q[10];
ry(2.7646012644972) q[11];
cx q[10],q[11];
ry(-2.9524864905864776) q[12];
ry(-0.4037235264974216) q[13];
cx q[12],q[13];
ry(0.3515190138771443) q[12];
ry(-0.9939195924124031) q[13];
cx q[12],q[13];
ry(1.3699764803944905) q[14];
ry(-1.0423685717790976) q[15];
cx q[14],q[15];
ry(0.15036075113116798) q[14];
ry(0.027318734864235417) q[15];
cx q[14],q[15];
ry(-0.5329401341470563) q[0];
ry(-0.364173793255671) q[2];
cx q[0],q[2];
ry(2.719534421043869) q[0];
ry(-2.218659890656686) q[2];
cx q[0],q[2];
ry(1.6558829333411147) q[2];
ry(0.3392526170493736) q[4];
cx q[2],q[4];
ry(3.1414552179798934) q[2];
ry(-0.00010572423581489687) q[4];
cx q[2],q[4];
ry(0.25211215321063474) q[4];
ry(-1.4121011866427904) q[6];
cx q[4],q[6];
ry(-0.22521334216464742) q[4];
ry(2.9612720948608464) q[6];
cx q[4],q[6];
ry(1.787290596806817) q[6];
ry(-1.776646313387472) q[8];
cx q[6],q[8];
ry(-3.0778541027748867) q[6];
ry(2.766631724503771) q[8];
cx q[6],q[8];
ry(1.8638234180708062) q[8];
ry(0.8893782681812583) q[10];
cx q[8],q[10];
ry(-0.3129025912185073) q[8];
ry(2.8884624942064887) q[10];
cx q[8],q[10];
ry(1.7230945271073133) q[10];
ry(2.0841422641559126) q[12];
cx q[10],q[12];
ry(0.008215784428932508) q[10];
ry(-3.1407447118799685) q[12];
cx q[10],q[12];
ry(0.15205973990517307) q[12];
ry(-0.9574021700114939) q[14];
cx q[12],q[14];
ry(0.21557170689459404) q[12];
ry(0.5046701917984724) q[14];
cx q[12],q[14];
ry(2.505489534847715) q[1];
ry(0.9260356236122771) q[3];
cx q[1],q[3];
ry(-0.0529761097939991) q[1];
ry(0.7508407753261691) q[3];
cx q[1],q[3];
ry(-0.38755749571024234) q[3];
ry(1.006881195126715) q[5];
cx q[3],q[5];
ry(-0.5295477360719115) q[3];
ry(3.1415283879861695) q[5];
cx q[3],q[5];
ry(-1.0508141976596086) q[5];
ry(1.2746495707039038) q[7];
cx q[5],q[7];
ry(-2.562729674399132) q[5];
ry(1.7707418306166627) q[7];
cx q[5],q[7];
ry(-0.40068463958625156) q[7];
ry(-3.0271617278499434) q[9];
cx q[7],q[9];
ry(-3.0583936701800214) q[7];
ry(1.7463584993578367) q[9];
cx q[7],q[9];
ry(-1.7389521085196558) q[9];
ry(-0.25312708916639126) q[11];
cx q[9],q[11];
ry(-0.8945295037217945) q[9];
ry(3.107108471728213) q[11];
cx q[9],q[11];
ry(-0.19916948548343605) q[11];
ry(-3.060385969283248) q[13];
cx q[11],q[13];
ry(-3.1342355584002544) q[11];
ry(0.007529933740443972) q[13];
cx q[11],q[13];
ry(-0.5341483501905318) q[13];
ry(2.830682524459495) q[15];
cx q[13],q[15];
ry(-2.423430691666148) q[13];
ry(0.09973336732888338) q[15];
cx q[13],q[15];
ry(2.399452287359785) q[0];
ry(-1.0002034772811232) q[1];
cx q[0],q[1];
ry(2.078804219409834) q[0];
ry(1.4324778608110318) q[1];
cx q[0],q[1];
ry(1.9064316528346568) q[2];
ry(2.893497784706304) q[3];
cx q[2],q[3];
ry(-2.182279989096754) q[2];
ry(-2.279407872209598) q[3];
cx q[2],q[3];
ry(1.520436570637253) q[4];
ry(2.8890980273314892) q[5];
cx q[4],q[5];
ry(-2.7733381416420104) q[4];
ry(-0.37051718036019465) q[5];
cx q[4],q[5];
ry(1.2277274279721224) q[6];
ry(-1.0080517894854795) q[7];
cx q[6],q[7];
ry(2.714797204385321) q[6];
ry(-0.35234175592337635) q[7];
cx q[6],q[7];
ry(0.7043336727633518) q[8];
ry(1.2965580707717193) q[9];
cx q[8],q[9];
ry(0.7827885119156536) q[8];
ry(-2.9667796514587406) q[9];
cx q[8],q[9];
ry(2.1888063725779494) q[10];
ry(-1.4157308115591256) q[11];
cx q[10],q[11];
ry(-1.3528568716208136) q[10];
ry(-1.7709485564493526) q[11];
cx q[10],q[11];
ry(-2.7530722731567865) q[12];
ry(-2.2340338855575395) q[13];
cx q[12],q[13];
ry(2.8412798475238485) q[12];
ry(1.2340062687447462) q[13];
cx q[12],q[13];
ry(-0.6036432591966625) q[14];
ry(-2.9113036557025285) q[15];
cx q[14],q[15];
ry(2.285834082566524) q[14];
ry(2.918706852443241) q[15];
cx q[14],q[15];
ry(-0.3079511870459764) q[0];
ry(-2.5384211727981283) q[2];
cx q[0],q[2];
ry(-0.8264111392452235) q[0];
ry(2.1723281725212678) q[2];
cx q[0],q[2];
ry(-0.0986089342364247) q[2];
ry(3.0102846599556643) q[4];
cx q[2],q[4];
ry(-1.1465807811252413) q[2];
ry(2.6892390239236165e-05) q[4];
cx q[2],q[4];
ry(-3.1097014125814617) q[4];
ry(1.0570160286583468) q[6];
cx q[4],q[6];
ry(1.5247846846177011) q[4];
ry(-1.6413063293578034) q[6];
cx q[4],q[6];
ry(-3.0603507947951814) q[6];
ry(-0.37455606376916006) q[8];
cx q[6],q[8];
ry(3.105844444522212) q[6];
ry(2.4517168839311707) q[8];
cx q[6],q[8];
ry(-0.6815969764042095) q[8];
ry(-0.6507010056644651) q[10];
cx q[8],q[10];
ry(-2.035813889203277) q[8];
ry(-2.4900495206727973) q[10];
cx q[8],q[10];
ry(2.4396694910805863) q[10];
ry(2.211447561496005) q[12];
cx q[10],q[12];
ry(-3.1386989955931206) q[10];
ry(-3.1402360347708655) q[12];
cx q[10],q[12];
ry(1.9245542867331744) q[12];
ry(2.106308891763312) q[14];
cx q[12],q[14];
ry(-0.9222830052671327) q[12];
ry(0.45157426280536955) q[14];
cx q[12],q[14];
ry(1.8690448975250877) q[1];
ry(0.06542579889414443) q[3];
cx q[1],q[3];
ry(-1.2968242172738655) q[1];
ry(0.9842287348720254) q[3];
cx q[1],q[3];
ry(2.5942037937598674) q[3];
ry(-1.6096394064125952) q[5];
cx q[3],q[5];
ry(3.1286956705284554) q[3];
ry(-3.1413438283324258) q[5];
cx q[3],q[5];
ry(0.5609492474806927) q[5];
ry(1.3025384031581755) q[7];
cx q[5],q[7];
ry(-2.1763161988341655) q[5];
ry(-1.3443148879056876) q[7];
cx q[5],q[7];
ry(0.0008497411531163124) q[7];
ry(-0.4171401617656717) q[9];
cx q[7],q[9];
ry(-0.0019810461459091044) q[7];
ry(-2.995873565499826) q[9];
cx q[7],q[9];
ry(0.5521325195139892) q[9];
ry(2.492497493621634) q[11];
cx q[9],q[11];
ry(2.9469595909937354) q[9];
ry(-0.24709798675013137) q[11];
cx q[9],q[11];
ry(1.1107696537931733) q[11];
ry(-2.0343718708860754) q[13];
cx q[11],q[13];
ry(0.010286132792197478) q[11];
ry(2.9725309429163285) q[13];
cx q[11],q[13];
ry(0.0739502647298041) q[13];
ry(-0.9015104880280385) q[15];
cx q[13],q[15];
ry(2.870023060911137) q[13];
ry(-0.012915051381395162) q[15];
cx q[13],q[15];
ry(-1.6009926430789716) q[0];
ry(0.3762485120729657) q[1];
cx q[0],q[1];
ry(-0.08377619199972397) q[0];
ry(3.0788355890553523) q[1];
cx q[0],q[1];
ry(0.7433027805287278) q[2];
ry(-0.7671253717829423) q[3];
cx q[2],q[3];
ry(0.9200818556461707) q[2];
ry(1.394947292441354) q[3];
cx q[2],q[3];
ry(2.259683995624556) q[4];
ry(-1.496679912760045) q[5];
cx q[4],q[5];
ry(-0.6265167903293832) q[4];
ry(-1.8007660268755377) q[5];
cx q[4],q[5];
ry(0.3254035149668789) q[6];
ry(-1.4899683471150986) q[7];
cx q[6],q[7];
ry(0.2167947465700021) q[6];
ry(0.42530835696465363) q[7];
cx q[6],q[7];
ry(-2.1076032197589516) q[8];
ry(-1.794745146024508) q[9];
cx q[8],q[9];
ry(-2.0460352133400908) q[8];
ry(-2.1461962745711554) q[9];
cx q[8],q[9];
ry(1.35581040520725) q[10];
ry(-2.002213445137147) q[11];
cx q[10],q[11];
ry(0.004418377502382498) q[10];
ry(0.04191908736955272) q[11];
cx q[10],q[11];
ry(0.6864318070219557) q[12];
ry(2.994253491937953) q[13];
cx q[12],q[13];
ry(-2.7463893350085327) q[12];
ry(2.234277045924714) q[13];
cx q[12],q[13];
ry(-1.4899670577660622) q[14];
ry(0.9477414578225141) q[15];
cx q[14],q[15];
ry(-2.1639071635252036) q[14];
ry(-2.1114648720618114) q[15];
cx q[14],q[15];
ry(-0.07118924258264267) q[0];
ry(-0.37608301767933305) q[2];
cx q[0],q[2];
ry(-1.4990215641774345) q[0];
ry(-1.9163437767218223) q[2];
cx q[0],q[2];
ry(2.6191541781195524) q[2];
ry(-1.3461225633047647) q[4];
cx q[2],q[4];
ry(1.4200600494600286) q[2];
ry(-0.43001641086105913) q[4];
cx q[2],q[4];
ry(-2.112924969841366) q[4];
ry(-1.6025819814779023) q[6];
cx q[4],q[6];
ry(0.0024965376108467534) q[4];
ry(-3.1413819351813324) q[6];
cx q[4],q[6];
ry(1.4957654066216026) q[6];
ry(1.0803502279027768) q[8];
cx q[6],q[8];
ry(3.0910678251864785) q[6];
ry(-1.9460123101021827) q[8];
cx q[6],q[8];
ry(1.9265643750066022) q[8];
ry(1.0212083234362748) q[10];
cx q[8],q[10];
ry(-0.9335232574887184) q[8];
ry(2.5609259327017755) q[10];
cx q[8],q[10];
ry(0.4627625712490868) q[10];
ry(-2.0151152944312454) q[12];
cx q[10],q[12];
ry(3.1157930229884863) q[10];
ry(-0.006289687632837774) q[12];
cx q[10],q[12];
ry(-0.9012131893397715) q[12];
ry(2.648362326834359) q[14];
cx q[12],q[14];
ry(-2.8987159649811884) q[12];
ry(-0.07218542863702826) q[14];
cx q[12],q[14];
ry(0.06435862214340868) q[1];
ry(2.6199048673295318) q[3];
cx q[1],q[3];
ry(-1.4989305964272288) q[1];
ry(2.2284777959449475) q[3];
cx q[1],q[3];
ry(0.5881536762420777) q[3];
ry(1.1533310277776316) q[5];
cx q[3],q[5];
ry(-1.998239685926376) q[3];
ry(-3.1047316541172822) q[5];
cx q[3],q[5];
ry(-2.2517783525137225) q[5];
ry(0.24275430579062796) q[7];
cx q[5],q[7];
ry(0.016880752589790146) q[5];
ry(3.1233231187872437) q[7];
cx q[5],q[7];
ry(-1.5926645613317714) q[7];
ry(-0.73545990177864) q[9];
cx q[7],q[9];
ry(-3.011987910189023) q[7];
ry(3.133970805456757) q[9];
cx q[7],q[9];
ry(1.447442571514209) q[9];
ry(-1.5139668904485508) q[11];
cx q[9],q[11];
ry(-3.1288120604193583) q[9];
ry(-0.039119368649716435) q[11];
cx q[9],q[11];
ry(2.8445122295789376) q[11];
ry(1.4192860440391752) q[13];
cx q[11],q[13];
ry(3.113688692668575) q[11];
ry(-2.9308769224967466) q[13];
cx q[11],q[13];
ry(2.140409661985041) q[13];
ry(-0.041598923858566256) q[15];
cx q[13],q[15];
ry(3.034131218945833) q[13];
ry(3.1146960827922925) q[15];
cx q[13],q[15];
ry(-0.47704836546646295) q[0];
ry(2.2778355538897292) q[1];
cx q[0],q[1];
ry(-2.973185419445811) q[0];
ry(-1.0883244062654929) q[1];
cx q[0],q[1];
ry(1.655737788721533) q[2];
ry(2.192604318987903) q[3];
cx q[2],q[3];
ry(-3.1072675247640302) q[2];
ry(1.221447696771933) q[3];
cx q[2],q[3];
ry(-0.41234094015437456) q[4];
ry(0.890372035316478) q[5];
cx q[4],q[5];
ry(-2.288514217996474) q[4];
ry(-1.3210222339096296) q[5];
cx q[4],q[5];
ry(1.3312740008091133) q[6];
ry(-2.2772080602932467) q[7];
cx q[6],q[7];
ry(2.657408159179334) q[6];
ry(1.277233722892725) q[7];
cx q[6],q[7];
ry(-2.4144596690331364) q[8];
ry(-2.9433875725777705) q[9];
cx q[8],q[9];
ry(-2.469352155677028) q[8];
ry(2.2723615531312182) q[9];
cx q[8],q[9];
ry(1.0139731634740814) q[10];
ry(2.732582703363317) q[11];
cx q[10],q[11];
ry(1.283137678952854) q[10];
ry(3.0872523632450326) q[11];
cx q[10],q[11];
ry(-1.0127452375399812) q[12];
ry(0.23232226178619353) q[13];
cx q[12],q[13];
ry(-0.8676142458838134) q[12];
ry(0.17649274242512902) q[13];
cx q[12],q[13];
ry(0.9842523525384683) q[14];
ry(-0.8673709653037687) q[15];
cx q[14],q[15];
ry(-2.8019727527756135) q[14];
ry(2.307435013731131) q[15];
cx q[14],q[15];
ry(2.9255235848769092) q[0];
ry(2.843356613398456) q[2];
cx q[0],q[2];
ry(-3.11807209129057) q[0];
ry(3.1196011933656207) q[2];
cx q[0],q[2];
ry(-2.0901405481941113) q[2];
ry(-2.8885252896815574) q[4];
cx q[2],q[4];
ry(0.1430158039572116) q[2];
ry(-3.062936528157347) q[4];
cx q[2],q[4];
ry(2.8336581534990737) q[4];
ry(1.970104221967599) q[6];
cx q[4],q[6];
ry(-0.04922736207027878) q[4];
ry(0.2461648188744805) q[6];
cx q[4],q[6];
ry(-2.7872103908929) q[6];
ry(-1.5958203401963047) q[8];
cx q[6],q[8];
ry(-3.0907151473175) q[6];
ry(3.1122404725838844) q[8];
cx q[6],q[8];
ry(1.5390771766137736) q[8];
ry(-1.1459841761640037) q[10];
cx q[8],q[10];
ry(0.007381275055484693) q[8];
ry(0.019269700263766225) q[10];
cx q[8],q[10];
ry(2.7232971860957793) q[10];
ry(2.2669371453995573) q[12];
cx q[10],q[12];
ry(-3.1381915103109748) q[10];
ry(-0.012123563204825771) q[12];
cx q[10],q[12];
ry(2.452254754713854) q[12];
ry(0.3504523406518034) q[14];
cx q[12],q[14];
ry(-0.12644911215806043) q[12];
ry(0.10182331879060234) q[14];
cx q[12],q[14];
ry(-0.04588307170134219) q[1];
ry(-2.0607886920329364) q[3];
cx q[1],q[3];
ry(0.0030927971461700565) q[1];
ry(1.5951796929962505) q[3];
cx q[1],q[3];
ry(-0.19684603685399746) q[3];
ry(2.916503270090764) q[5];
cx q[3],q[5];
ry(1.996597680431929) q[3];
ry(-3.1247646282393604) q[5];
cx q[3],q[5];
ry(-2.6281871976591726) q[5];
ry(2.0711772763753653) q[7];
cx q[5],q[7];
ry(0.011445534859447009) q[5];
ry(-3.13800512279908) q[7];
cx q[5],q[7];
ry(2.81043851881188) q[7];
ry(1.1696052876321057) q[9];
cx q[7],q[9];
ry(3.0291388145574176) q[7];
ry(0.007492386112510907) q[9];
cx q[7],q[9];
ry(1.9632571896196525) q[9];
ry(-0.10192209893335047) q[11];
cx q[9],q[11];
ry(0.03579427974793398) q[9];
ry(-1.7663801923726785) q[11];
cx q[9],q[11];
ry(3.1252259581754314) q[11];
ry(-2.0056080115121535) q[13];
cx q[11],q[13];
ry(2.1204550132640096) q[11];
ry(0.07044031232885488) q[13];
cx q[11],q[13];
ry(-1.9156448931667218) q[13];
ry(2.841509061580835) q[15];
cx q[13],q[15];
ry(-0.10251117206166947) q[13];
ry(-3.133881100948944) q[15];
cx q[13],q[15];
ry(1.8869022009611083) q[0];
ry(0.7547663153377382) q[1];
cx q[0],q[1];
ry(-3.051772705860113) q[0];
ry(2.8834730515646814) q[1];
cx q[0],q[1];
ry(-3.067521566720418) q[2];
ry(2.0899505604375657) q[3];
cx q[2],q[3];
ry(0.19781421661377685) q[2];
ry(-1.7932077713167682) q[3];
cx q[2],q[3];
ry(2.9094699614748185) q[4];
ry(-0.9860441224933295) q[5];
cx q[4],q[5];
ry(-1.69627618789651) q[4];
ry(1.569438873121447) q[5];
cx q[4],q[5];
ry(-2.899587452155653) q[6];
ry(-2.583163465613924) q[7];
cx q[6],q[7];
ry(2.18945146030074) q[6];
ry(1.91072546577632) q[7];
cx q[6],q[7];
ry(1.4852879730041306) q[8];
ry(-1.5762760422791384) q[9];
cx q[8],q[9];
ry(-2.888256241250122) q[8];
ry(-3.0878737655788786) q[9];
cx q[8],q[9];
ry(-1.108288940783683) q[10];
ry(-3.028820873663809) q[11];
cx q[10],q[11];
ry(-2.375956208746285) q[10];
ry(-2.889339618533026) q[11];
cx q[10],q[11];
ry(0.4303404092184664) q[12];
ry(-0.6567275783162732) q[13];
cx q[12],q[13];
ry(-2.93620155842047) q[12];
ry(-1.820145224800937) q[13];
cx q[12],q[13];
ry(-2.904306994104067) q[14];
ry(-1.5804812941251356) q[15];
cx q[14],q[15];
ry(0.02306047800191724) q[14];
ry(0.4388153128441435) q[15];
cx q[14],q[15];
ry(1.1865569551237467) q[0];
ry(0.7833141639284742) q[2];
cx q[0],q[2];
ry(-3.135799764817409) q[0];
ry(0.1321475788962072) q[2];
cx q[0],q[2];
ry(-0.6765999252247523) q[2];
ry(0.8519740005888039) q[4];
cx q[2],q[4];
ry(1.8648436492337046) q[2];
ry(-0.1736380651987961) q[4];
cx q[2],q[4];
ry(-1.0653265047260125) q[4];
ry(-1.9141059934386293) q[6];
cx q[4],q[6];
ry(3.141314146807458) q[4];
ry(-0.005556866481007283) q[6];
cx q[4],q[6];
ry(1.0536514005530098) q[6];
ry(2.6220415351634516) q[8];
cx q[6],q[8];
ry(-3.0879119915599262) q[6];
ry(3.0859216926036086) q[8];
cx q[6],q[8];
ry(2.3376860530613826) q[8];
ry(3.0535455215899034) q[10];
cx q[8],q[10];
ry(-3.099364350435147) q[8];
ry(3.103050617858782) q[10];
cx q[8],q[10];
ry(-1.0099773531212293) q[10];
ry(2.8357208538292547) q[12];
cx q[10],q[12];
ry(-3.115839247939319) q[10];
ry(0.017570702147588158) q[12];
cx q[10],q[12];
ry(0.9266325774901024) q[12];
ry(0.6252162171553204) q[14];
cx q[12],q[14];
ry(1.2749700872557843) q[12];
ry(2.361400978621291) q[14];
cx q[12],q[14];
ry(-2.2486706299047188) q[1];
ry(-1.99651563085766) q[3];
cx q[1],q[3];
ry(0.0587338281750787) q[1];
ry(-0.8327293236007014) q[3];
cx q[1],q[3];
ry(2.3081417629899073) q[3];
ry(0.5114668131430792) q[5];
cx q[3],q[5];
ry(3.1343776785343644) q[3];
ry(3.12862603422108) q[5];
cx q[3],q[5];
ry(-0.6802427783670071) q[5];
ry(-3.1011345857086803) q[7];
cx q[5],q[7];
ry(0.10330701888901622) q[5];
ry(-3.135871197368038) q[7];
cx q[5],q[7];
ry(-2.801915853884468) q[7];
ry(-1.1577508496785052) q[9];
cx q[7],q[9];
ry(3.0840920714396987) q[7];
ry(0.2124062101809178) q[9];
cx q[7],q[9];
ry(-2.292864056118609) q[9];
ry(-1.681955555788375) q[11];
cx q[9],q[11];
ry(-0.031088476182290607) q[9];
ry(0.0004162916623622337) q[11];
cx q[9],q[11];
ry(1.9816386939764594) q[11];
ry(2.1041638337247877) q[13];
cx q[11],q[13];
ry(-0.010170767904639002) q[11];
ry(-3.0704627361568955) q[13];
cx q[11],q[13];
ry(-1.0071450468518979) q[13];
ry(0.647623233617165) q[15];
cx q[13],q[15];
ry(-0.289056870764873) q[13];
ry(-3.0638892703530582) q[15];
cx q[13],q[15];
ry(-1.0399816225834286) q[0];
ry(0.16102100992825233) q[1];
cx q[0],q[1];
ry(-1.5749270715236559) q[0];
ry(-2.8578211161670803) q[1];
cx q[0],q[1];
ry(-1.73902767531317) q[2];
ry(0.06371056764502123) q[3];
cx q[2],q[3];
ry(3.1323324649726345) q[2];
ry(3.068487138219063) q[3];
cx q[2],q[3];
ry(-3.1037329286088653) q[4];
ry(-0.4527147200748889) q[5];
cx q[4],q[5];
ry(-2.9710723133037704) q[4];
ry(3.125075829211698) q[5];
cx q[4],q[5];
ry(-1.8141696502909026) q[6];
ry(3.0598212724627953) q[7];
cx q[6],q[7];
ry(-0.018118938482501346) q[6];
ry(-0.04153960810554658) q[7];
cx q[6],q[7];
ry(1.3254554616920824) q[8];
ry(1.31116938048988) q[9];
cx q[8],q[9];
ry(2.909018887985397) q[8];
ry(1.5508915198820103) q[9];
cx q[8],q[9];
ry(-0.5396947186667188) q[10];
ry(1.4092574994546645) q[11];
cx q[10],q[11];
ry(2.7852140218384736) q[10];
ry(1.897863692271074) q[11];
cx q[10],q[11];
ry(0.7360901116071821) q[12];
ry(1.7757575322083738) q[13];
cx q[12],q[13];
ry(-0.02124126099621778) q[12];
ry(-0.04390367721278299) q[13];
cx q[12],q[13];
ry(-0.5644583643442768) q[14];
ry(-3.0408770290921887) q[15];
cx q[14],q[15];
ry(1.7202444484536608) q[14];
ry(0.0555487747013683) q[15];
cx q[14],q[15];
ry(-0.0787170152110921) q[0];
ry(1.58469287709572) q[2];
cx q[0],q[2];
ry(0.0032322506003461537) q[0];
ry(0.016677735986817575) q[2];
cx q[0],q[2];
ry(-0.157241050709235) q[2];
ry(2.747631546118717) q[4];
cx q[2],q[4];
ry(-2.287305544603024) q[2];
ry(-0.1763332251498291) q[4];
cx q[2],q[4];
ry(-0.2708033047492663) q[4];
ry(-3.0843169837157767) q[6];
cx q[4],q[6];
ry(-0.03472242883195964) q[4];
ry(0.008464178874523144) q[6];
cx q[4],q[6];
ry(-1.1496103418200845) q[6];
ry(1.5423882113579437) q[8];
cx q[6],q[8];
ry(-3.0833783740616334) q[6];
ry(3.1391399616280444) q[8];
cx q[6],q[8];
ry(1.3232569189216512) q[8];
ry(0.4353488229286996) q[10];
cx q[8],q[10];
ry(-5.2944802857446405e-05) q[8];
ry(-3.1414974658762183) q[10];
cx q[8],q[10];
ry(-1.8096223186349736) q[10];
ry(0.561496083529935) q[12];
cx q[10],q[12];
ry(3.1396023107633924) q[10];
ry(-3.1387537483182757) q[12];
cx q[10],q[12];
ry(-1.9395602646281382) q[12];
ry(0.5518043057413575) q[14];
cx q[12],q[14];
ry(2.6776290688613242) q[12];
ry(2.338668562694306) q[14];
cx q[12],q[14];
ry(-2.6330703731049927) q[1];
ry(2.9111423530207645) q[3];
cx q[1],q[3];
ry(3.127596979517005) q[1];
ry(2.9104932550883262) q[3];
cx q[1],q[3];
ry(2.8864253789466905) q[3];
ry(-2.46069293732329) q[5];
cx q[3],q[5];
ry(-0.010247249794399949) q[3];
ry(-3.1346515748035912) q[5];
cx q[3],q[5];
ry(-2.5424537593041525) q[5];
ry(1.603697446000092) q[7];
cx q[5],q[7];
ry(-0.001637116742621557) q[5];
ry(-0.0025026885905478384) q[7];
cx q[5],q[7];
ry(-1.5332510510898916) q[7];
ry(-2.4481364490559567) q[9];
cx q[7],q[9];
ry(3.131371345460734) q[7];
ry(-0.20372553593198375) q[9];
cx q[7],q[9];
ry(2.3813106495550618) q[9];
ry(-1.8356492761638532) q[11];
cx q[9],q[11];
ry(-3.1207117540030764) q[9];
ry(3.0985222298539568) q[11];
cx q[9],q[11];
ry(-0.4139481866837987) q[11];
ry(-2.8716981759101623) q[13];
cx q[11],q[13];
ry(-3.1412874016328227) q[11];
ry(0.014251683692993334) q[13];
cx q[11],q[13];
ry(0.9604395969264384) q[13];
ry(-3.022188782342532) q[15];
cx q[13],q[15];
ry(-0.22260215196627353) q[13];
ry(-3.1394501593785544) q[15];
cx q[13],q[15];
ry(-0.03697503785697708) q[0];
ry(0.877733926599058) q[1];
ry(-0.5220317652831543) q[2];
ry(1.7420650282340224) q[3];
ry(-0.9362751303532688) q[4];
ry(-2.2022892567274104) q[5];
ry(-0.5116536549311217) q[6];
ry(0.24061637601630692) q[7];
ry(-1.555978398052976) q[8];
ry(0.5590204808270407) q[9];
ry(0.739344671665104) q[10];
ry(2.943136444807637) q[11];
ry(-2.893216617878774) q[12];
ry(2.11789239786956) q[13];
ry(-3.0576903763176326) q[14];
ry(1.5050131336188475) q[15];