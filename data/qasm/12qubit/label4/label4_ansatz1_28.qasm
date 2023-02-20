OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.9693204774706146) q[0];
rz(0.4430999598713924) q[0];
ry(2.274312893242622) q[1];
rz(2.011760808807465) q[1];
ry(-2.2015500636688525) q[2];
rz(1.7363217635829882) q[2];
ry(-2.167444456935904) q[3];
rz(1.2346783002060708) q[3];
ry(0.9418710056035895) q[4];
rz(2.697184585322769) q[4];
ry(-1.8658587134927194) q[5];
rz(2.7394715016256272) q[5];
ry(-2.320085973630711) q[6];
rz(0.5318763403982347) q[6];
ry(-3.1410990699200654) q[7];
rz(2.0376091359718362) q[7];
ry(2.1006779751833067) q[8];
rz(1.2047175345912517) q[8];
ry(-3.111109557255422) q[9];
rz(0.9700248277912045) q[9];
ry(-2.519309662201691) q[10];
rz(0.5616477657402295) q[10];
ry(-1.4142323089150293) q[11];
rz(2.029394647762408) q[11];
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
ry(0.25526903985361293) q[0];
rz(-1.6339131095834787) q[0];
ry(-3.113410036539592) q[1];
rz(-0.8296658717423525) q[1];
ry(0.0622654809289056) q[2];
rz(-1.9021268283163286) q[2];
ry(3.0556069626034916) q[3];
rz(2.1567637007988343) q[3];
ry(0.5372356103084011) q[4];
rz(1.292260438020509) q[4];
ry(-1.0095870268574245) q[5];
rz(-2.042020508291885) q[5];
ry(-0.5185879717994751) q[6];
rz(2.8282270717866904) q[6];
ry(0.7090126901945599) q[7];
rz(-1.8606164676158377) q[7];
ry(-3.1340095736128246) q[8];
rz(-2.0269431403836795) q[8];
ry(3.1278446172088588) q[9];
rz(-1.6785273504112264) q[9];
ry(-1.8167130285630249) q[10];
rz(-0.34946003183164065) q[10];
ry(2.2004501958006015) q[11];
rz(-0.1404834551485119) q[11];
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
ry(-1.5787995077211567) q[0];
rz(-1.753590214610771) q[0];
ry(2.6726661094437882) q[1];
rz(-1.5165919578979405) q[1];
ry(2.2445864025552944) q[2];
rz(0.40541837702523154) q[2];
ry(2.1991590255217384) q[3];
rz(-1.8263705270821093) q[3];
ry(-1.276604284417913) q[4];
rz(0.45471837358852985) q[4];
ry(0.9826394598749286) q[5];
rz(2.243549787679969) q[5];
ry(3.136159147781029) q[6];
rz(-0.15794420020260524) q[6];
ry(-3.1396623266956065) q[7];
rz(-2.091538982889251) q[7];
ry(0.333161024014629) q[8];
rz(-2.616595172967743) q[8];
ry(2.9897793221100826) q[9];
rz(-0.15008514057934352) q[9];
ry(-1.332725633139576) q[10];
rz(-1.0240282756781862) q[10];
ry(1.3248465357759853) q[11];
rz(2.312789189539195) q[11];
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
ry(1.939227239347905) q[0];
rz(1.0486644481057006) q[0];
ry(0.700328245157404) q[1];
rz(2.961745602624669) q[1];
ry(-0.08298160974072653) q[2];
rz(-1.7295208514879228) q[2];
ry(-2.831218336386153) q[3];
rz(2.801681870735527) q[3];
ry(-2.7368021293394857) q[4];
rz(-3.057699253688647) q[4];
ry(-2.635895617017083) q[5];
rz(2.472897225397679) q[5];
ry(-1.1473246500112717) q[6];
rz(-2.9483551572400972) q[6];
ry(-1.1718314163599493) q[7];
rz(1.7832520973534267) q[7];
ry(-2.139757544729943) q[8];
rz(-2.0477133644874597) q[8];
ry(-0.025258281791167292) q[9];
rz(0.3083349254161903) q[9];
ry(-0.1843710914894583) q[10];
rz(1.9046697635426595) q[10];
ry(1.9264246680375774) q[11];
rz(1.5213889904166586) q[11];
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
ry(2.250329717310609) q[0];
rz(-1.9125588308842856) q[0];
ry(0.6302401138075869) q[1];
rz(-1.2791657589684933) q[1];
ry(-1.7656855711097048) q[2];
rz(-1.442727366051125) q[2];
ry(-2.6160329066560437) q[3];
rz(-2.6558725169067876) q[3];
ry(-1.7507253747298914) q[4];
rz(2.505500486891342) q[4];
ry(-1.4985903581839182) q[5];
rz(0.9399435801612875) q[5];
ry(0.0038246889509069852) q[6];
rz(0.579294912085439) q[6];
ry(0.013052793129911144) q[7];
rz(-3.0657927693831457) q[7];
ry(1.6216538535629192) q[8];
rz(1.4305098152108708) q[8];
ry(-0.5023448465854372) q[9];
rz(-2.1492977474608566) q[9];
ry(-1.958221016622015) q[10];
rz(1.002087817612346) q[10];
ry(-0.43372953912948997) q[11];
rz(-0.43282272325144694) q[11];
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
ry(1.0771721729146773) q[0];
rz(2.030112455029035) q[0];
ry(-0.10110887157492865) q[1];
rz(2.146189260878097) q[1];
ry(-0.8514686729940997) q[2];
rz(-1.7136341111140876) q[2];
ry(3.0986220470999744) q[3];
rz(-0.30599540776558065) q[3];
ry(2.383191855171451) q[4];
rz(0.8630863647140627) q[4];
ry(-0.8798783157207627) q[5];
rz(-2.7061386430520367) q[5];
ry(-0.4686093308272703) q[6];
rz(2.9306822886642414) q[6];
ry(-2.6696282076011153) q[7];
rz(-0.4042707478822905) q[7];
ry(0.7790222343741452) q[8];
rz(0.4472108761114643) q[8];
ry(3.139983225467239) q[9];
rz(-1.9848331237377828) q[9];
ry(0.26696321816771784) q[10];
rz(-1.4716388264407927) q[10];
ry(-2.5134511619988493) q[11];
rz(-2.010551900082448) q[11];
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
ry(0.7814972637883056) q[0];
rz(-2.924028477635098) q[0];
ry(1.6444865436468459) q[1];
rz(1.255574080700651) q[1];
ry(-1.5608693599567005) q[2];
rz(0.036958406356705546) q[2];
ry(0.014455891168136503) q[3];
rz(-1.8922149745944834) q[3];
ry(-0.0006398167993393315) q[4];
rz(1.541813169555945) q[4];
ry(-2.7351197368875835) q[5];
rz(-1.2398448990708708) q[5];
ry(3.133921022098476) q[6];
rz(0.7492615107597741) q[6];
ry(3.1382638344050386) q[7];
rz(-2.329538361070843) q[7];
ry(2.917210445575752) q[8];
rz(0.7361568324451394) q[8];
ry(-0.06517424639449043) q[9];
rz(-0.13303590665892265) q[9];
ry(2.188425930064477) q[10];
rz(-0.45728543593620646) q[10];
ry(-0.009392493918739753) q[11];
rz(2.240085751126678) q[11];
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
ry(-2.9646127408853435) q[0];
rz(0.559671457347604) q[0];
ry(2.843491956740546) q[1];
rz(-2.2965483832733797) q[1];
ry(0.19420503081269125) q[2];
rz(3.046128388654619) q[2];
ry(0.3422782419966099) q[3];
rz(-2.869786939664739) q[3];
ry(-0.871605867140853) q[4];
rz(2.201462771586584) q[4];
ry(1.4424433841923212) q[5];
rz(-2.044169791564615) q[5];
ry(-2.48905674793466) q[6];
rz(0.9354650022471117) q[6];
ry(-1.5296320248326887) q[7];
rz(-2.1089725144286655) q[7];
ry(0.7059507795521434) q[8];
rz(-1.5441679817646463) q[8];
ry(3.1389390137268616) q[9];
rz(-0.16970014454170873) q[9];
ry(-0.5169766853384967) q[10];
rz(-0.6242243008859063) q[10];
ry(-2.5053232677996276) q[11];
rz(1.444629487516782) q[11];
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
ry(-2.7656091096767947) q[0];
rz(-0.9822248032559932) q[0];
ry(0.11748330736028834) q[1];
rz(0.935057746550741) q[1];
ry(-2.8855179727516225) q[2];
rz(1.50060598478664) q[2];
ry(-1.1144108187375865) q[3];
rz(-0.2410296477997208) q[3];
ry(1.6712729721059691) q[4];
rz(2.6523610949579997) q[4];
ry(-2.1664354039315836) q[5];
rz(-0.9664310609499784) q[5];
ry(0.0003704077424400154) q[6];
rz(0.48586543754009487) q[6];
ry(2.623110266556377) q[7];
rz(-1.4436395279216807) q[7];
ry(1.6998909530442332) q[8];
rz(1.6661125053447794) q[8];
ry(2.0678980663003337) q[9];
rz(-2.7517734347971845) q[9];
ry(0.5420685547959039) q[10];
rz(1.096038870688428) q[10];
ry(-2.4168975647791053) q[11];
rz(-0.750019375912725) q[11];
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
ry(-0.008053072995676574) q[0];
rz(1.636000455272601) q[0];
ry(-2.233997118761846) q[1];
rz(-2.63623870141737) q[1];
ry(-0.18446351884000478) q[2];
rz(-0.34591398707258936) q[2];
ry(-1.6830046456110008) q[3];
rz(-1.177447824103905) q[3];
ry(-3.0289750066222227) q[4];
rz(3.0351026145101465) q[4];
ry(-2.973090657822703) q[5];
rz(-1.7491358272009137) q[5];
ry(-1.6161487211868844) q[6];
rz(2.205624633999852) q[6];
ry(-1.4160655236265276) q[7];
rz(-2.283145658841164) q[7];
ry(0.00614978355995266) q[8];
rz(-3.140779415434256) q[8];
ry(-1.496581341485141) q[9];
rz(1.8497224528733591) q[9];
ry(-1.3315339689313666) q[10];
rz(-3.08419941691757) q[10];
ry(-0.1646045760134484) q[11];
rz(0.3294432226766659) q[11];
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
ry(0.7205408734630532) q[0];
rz(-3.0937473895125573) q[0];
ry(0.6821641635762518) q[1];
rz(-0.40990557237020675) q[1];
ry(-2.1424036092662933) q[2];
rz(2.7653738108751567) q[2];
ry(-1.5148935143452797) q[3];
rz(-2.742083578426584) q[3];
ry(-0.7771987947986174) q[4];
rz(0.9239045361462903) q[4];
ry(1.578891117839634) q[5];
rz(-2.673390109095054) q[5];
ry(0.014806284304766557) q[6];
rz(0.9766464162764912) q[6];
ry(1.5086139066363309) q[7];
rz(1.5905173894268192) q[7];
ry(2.8503916726446032) q[8];
rz(-1.8152923988290104) q[8];
ry(-3.0152206800055805) q[9];
rz(-3.0378279966688595) q[9];
ry(-1.3897907366078193) q[10];
rz(2.704608172215521) q[10];
ry(-0.1538738386003592) q[11];
rz(-1.2666559008100657) q[11];
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
ry(3.050713692989968) q[0];
rz(-1.3155592519395416) q[0];
ry(2.8373736815988364) q[1];
rz(2.7447305645293376) q[1];
ry(-1.0681880620227666) q[2];
rz(1.85370160813768) q[2];
ry(-3.0519573303682606) q[3];
rz(-0.6811301609534381) q[3];
ry(-0.004652527046941302) q[4];
rz(-0.8050648965981003) q[4];
ry(-2.8414686593117517) q[5];
rz(0.49179497913366443) q[5];
ry(-1.6387326871630132) q[6];
rz(-0.45381215707662115) q[6];
ry(2.866002083434248) q[7];
rz(-2.1884750810193943) q[7];
ry(0.0028901976562961218) q[8];
rz(0.019586347875360736) q[8];
ry(0.6651988800818426) q[9];
rz(-2.7691041615885768) q[9];
ry(-1.6418138552541657) q[10];
rz(-0.14695372885163807) q[10];
ry(-0.04868543030761992) q[11];
rz(-2.116549485275264) q[11];
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
ry(-1.7924203945963182) q[0];
rz(2.191073177856509) q[0];
ry(0.8469826377589244) q[1];
rz(-0.7617245987309095) q[1];
ry(-2.0547212387454623) q[2];
rz(-2.54267645997133) q[2];
ry(2.151937354592973) q[3];
rz(0.17926091995320625) q[3];
ry(-0.05892404534758286) q[4];
rz(1.8843488478406671) q[4];
ry(2.964751620384113) q[5];
rz(1.2876183361567106) q[5];
ry(3.1358107979644423) q[6];
rz(2.045963994171342) q[6];
ry(0.9023165945890925) q[7];
rz(1.9928993762707723) q[7];
ry(-1.054649620235287) q[8];
rz(-0.8993004627725636) q[8];
ry(-1.017050826765356) q[9];
rz(-0.015784443413732367) q[9];
ry(1.8633858058736095) q[10];
rz(-1.166045644937416) q[10];
ry(3.06283231204278) q[11];
rz(-2.7698050242680083) q[11];
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
ry(-0.6561348085274314) q[0];
rz(2.680680578055374) q[0];
ry(-2.109905005350411) q[1];
rz(0.3128572243293269) q[1];
ry(2.562323342149834) q[2];
rz(1.7956121808136825) q[2];
ry(0.04425548031624604) q[3];
rz(-1.1160876606793548) q[3];
ry(-3.138982413595447) q[4];
rz(-0.4558371520110543) q[4];
ry(-1.7494873283105645) q[5];
rz(0.9631030392701311) q[5];
ry(0.1560893025193435) q[6];
rz(1.3460776454663141) q[6];
ry(-1.1651561581396113) q[7];
rz(-2.126093581673663) q[7];
ry(3.1334835986094762) q[8];
rz(-0.6342260431225837) q[8];
ry(3.1393825257728367) q[9];
rz(-2.4912882722879437) q[9];
ry(-2.3568931290396407) q[10];
rz(2.430247094073241) q[10];
ry(-3.061043900795664) q[11];
rz(3.1203382113038867) q[11];
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
ry(-1.9757386228842329) q[0];
rz(0.7524937349105658) q[0];
ry(1.3081014374107474) q[1];
rz(1.8076371523252597) q[1];
ry(2.784763450925759) q[2];
rz(-0.44980747933199167) q[2];
ry(0.8625384267899044) q[3];
rz(-2.344393297976229) q[3];
ry(-3.1357464359712175) q[4];
rz(-2.6825203600588132) q[4];
ry(-3.0977970712958336) q[5];
rz(-3.035829083813709) q[5];
ry(3.14087557313672) q[6];
rz(1.070737335479654) q[6];
ry(2.340522381762452) q[7];
rz(1.501008231873488) q[7];
ry(-2.1928918954877403) q[8];
rz(-2.30021811454928) q[8];
ry(0.3520247063380211) q[9];
rz(-2.252360113565958) q[9];
ry(3.0866660150317515) q[10];
rz(-0.7627576836247191) q[10];
ry(-0.4104613643998803) q[11];
rz(2.300097527713652) q[11];
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
ry(1.7976402874198394) q[0];
rz(-2.9599111975242876) q[0];
ry(2.8456299600238744) q[1];
rz(-0.9074355005921457) q[1];
ry(-1.808581782127078) q[2];
rz(1.9539528382570437) q[2];
ry(-3.119487769415603) q[3];
rz(1.0905070616116395) q[3];
ry(0.006632696068226807) q[4];
rz(2.256986557468341) q[4];
ry(-1.4944776767349417) q[5];
rz(-2.246011571561654) q[5];
ry(-3.1285061794503557) q[6];
rz(3.0856863285062595) q[6];
ry(0.5291873194628273) q[7];
rz(2.253006482676187) q[7];
ry(-3.1414866948857845) q[8];
rz(0.8610026444027747) q[8];
ry(0.8993531935302044) q[9];
rz(3.1236530160302416) q[9];
ry(1.2105939855118653) q[10];
rz(1.6687321459310995) q[10];
ry(0.14914818586794087) q[11];
rz(2.4172851887139806) q[11];
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
ry(-1.3889994146198807) q[0];
rz(-1.4310872494296358) q[0];
ry(2.8218979052391915) q[1];
rz(2.0632481722662064) q[1];
ry(1.620471633070549) q[2];
rz(-2.7451311824788704) q[2];
ry(0.29960332570607595) q[3];
rz(0.49520232754819005) q[3];
ry(2.62766374028907) q[4];
rz(-1.0237266845128232) q[4];
ry(0.06181529212681423) q[5];
rz(-2.2985585068141434) q[5];
ry(-3.1378904632314013) q[6];
rz(-2.0426627345440975) q[6];
ry(-0.9948808239407178) q[7];
rz(0.36638047737785856) q[7];
ry(1.032728797906456) q[8];
rz(-0.08845329176972162) q[8];
ry(-1.6933510860089385) q[9];
rz(0.8765851918569415) q[9];
ry(-0.20134903212314817) q[10];
rz(-0.8555648392261145) q[10];
ry(0.5077311565131958) q[11];
rz(0.3696748910740629) q[11];
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
ry(3.0789390587756658) q[0];
rz(-1.1492013292647247) q[0];
ry(1.6386828021260875) q[1];
rz(-2.0621624524640225) q[1];
ry(-0.7581301738848353) q[2];
rz(-0.01867791767083343) q[2];
ry(0.0024414896540196906) q[3];
rz(0.8136096055585593) q[3];
ry(3.1178732837592373) q[4];
rz(-0.2671653580405398) q[4];
ry(-3.135662295095354) q[5];
rz(0.728818442868973) q[5];
ry(-1.6530627564142122) q[6];
rz(-2.7838965149231965) q[6];
ry(-0.10507023996178497) q[7];
rz(0.6796720619350799) q[7];
ry(1.5618777500418854) q[8];
rz(0.08643670207365049) q[8];
ry(2.7576428501221075) q[9];
rz(-1.2185056507638619) q[9];
ry(0.5649433150018863) q[10];
rz(2.7934956250248133) q[10];
ry(-3.120832997945831) q[11];
rz(-2.7998281105052056) q[11];
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
ry(-0.8544119241474617) q[0];
rz(1.8804953210622197) q[0];
ry(1.3284954160282076) q[1];
rz(0.5506011231306278) q[1];
ry(1.4482458754475624) q[2];
rz(3.1172469741814264) q[2];
ry(3.1367298416896547) q[3];
rz(0.2994694146490405) q[3];
ry(2.01695692582513) q[4];
rz(2.2088954108749617) q[4];
ry(3.107224618582309) q[5];
rz(-3.0425262294594093) q[5];
ry(0.004446118791171294) q[6];
rz(-1.807489832038682) q[6];
ry(-0.10304663552789045) q[7];
rz(0.3803865329794378) q[7];
ry(-0.48786602555198133) q[8];
rz(1.8426850053909911) q[8];
ry(3.141175349846138) q[9];
rz(-0.8952445575813943) q[9];
ry(3.1302110697517658) q[10];
rz(2.616057878759835) q[10];
ry(-2.537030010698927) q[11];
rz(-2.7822842947706397) q[11];
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
ry(-1.7464122224056373) q[0];
rz(-3.017424894366698) q[0];
ry(2.299472794221849) q[1];
rz(0.7775156889683438) q[1];
ry(-1.9035015089291705) q[2];
rz(-3.0175726052325995) q[2];
ry(1.7126271805669693) q[3];
rz(0.3675237502719222) q[3];
ry(-1.5902388160836831) q[4];
rz(-0.03569812702465144) q[4];
ry(-0.7569817342096705) q[5];
rz(3.0851537483303386) q[5];
ry(-3.106383675955223) q[6];
rz(0.03544077474300325) q[6];
ry(3.1193797851477405) q[7];
rz(2.8984586405449595) q[7];
ry(0.027988723617199) q[8];
rz(1.2788323419672898) q[8];
ry(-0.12906437810882476) q[9];
rz(-0.9772062581552454) q[9];
ry(-0.6022475694826527) q[10];
rz(-0.7390712943257354) q[10];
ry(-3.109997623711503) q[11];
rz(-0.9247603193470902) q[11];
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
ry(-1.792436162901945) q[0];
rz(-1.5040885723909891) q[0];
ry(-1.3009137415015497) q[1];
rz(1.2271842300941165) q[1];
ry(0.01952831337158085) q[2];
rz(-0.2895182805573216) q[2];
ry(-3.1313071753003463) q[3];
rz(-2.784966769060514) q[3];
ry(2.642869259504312) q[4];
rz(1.5197133592203882) q[4];
ry(-0.3690153982789859) q[5];
rz(0.0012549150363003747) q[5];
ry(-3.060303839523311) q[6];
rz(1.128652394807478) q[6];
ry(0.005082464863551757) q[7];
rz(1.7868851966791046) q[7];
ry(-2.5574450539478155) q[8];
rz(-0.4337091569333657) q[8];
ry(1.5860393070052623) q[9];
rz(0.0017336837669104101) q[9];
ry(-1.7484719246840994) q[10];
rz(0.0029253666506896536) q[10];
ry(-0.4371142610969798) q[11];
rz(2.1441003301292656) q[11];
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
ry(1.7311382856177948) q[0];
rz(2.566725740899335) q[0];
ry(-2.1431658667849853) q[1];
rz(1.5740166315399193) q[1];
ry(-1.2435156114888744) q[2];
rz(-1.281645624967377) q[2];
ry(0.9078538660242559) q[3];
rz(-0.6703375335859025) q[3];
ry(-0.01976509265725834) q[4];
rz(-2.809835213503764) q[4];
ry(-3.034278795127841) q[5];
rz(3.0624535868781373) q[5];
ry(0.05705764746056374) q[6];
rz(2.077056815998852) q[6];
ry(0.007471378943114186) q[7];
rz(-0.9055260091715001) q[7];
ry(0.004337985898176022) q[8];
rz(-2.5758015456027503) q[8];
ry(-1.733781340241992) q[9];
rz(-3.1398838036380923) q[9];
ry(1.5637583850940242) q[10];
rz(-2.5516173726855405) q[10];
ry(-3.0947944159393925) q[11];
rz(-0.6771759351329191) q[11];
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
ry(0.09333746607266082) q[0];
rz(2.272867545858279) q[0];
ry(-0.6984064713608706) q[1];
rz(-0.19371875756709486) q[1];
ry(-1.7762950300996965) q[2];
rz(0.025148010033361458) q[2];
ry(-3.133060753247681) q[3];
rz(-1.3955586296957065) q[3];
ry(3.14057531344267) q[4];
rz(2.9881820771036107) q[4];
ry(-1.9426963406919429) q[5];
rz(-1.1766200158717173) q[5];
ry(-3.050402717057654) q[6];
rz(-0.15101353149274033) q[6];
ry(0.2823793130709453) q[7];
rz(-3.029456517854245) q[7];
ry(-0.058829730511868863) q[8];
rz(-0.06445607640970064) q[8];
ry(-0.7184959571257776) q[9];
rz(-2.3344419217885215) q[9];
ry(0.03264885362004488) q[10];
rz(0.004310057240635382) q[10];
ry(-1.6644137745397658) q[11];
rz(1.5592127472115864) q[11];
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
ry(-2.0038056819461403) q[0];
rz(1.3063279934299077) q[0];
ry(-0.013195242910587266) q[1];
rz(0.04327300407092327) q[1];
ry(-1.1800131177984179) q[2];
rz(-2.1168076127119893) q[2];
ry(-3.1034274969237825) q[3];
rz(-2.274097254972161) q[3];
ry(-0.8224750115508428) q[4];
rz(2.8124472925547286) q[4];
ry(3.1112880415328883) q[5];
rz(-1.1261733549531376) q[5];
ry(-0.07186871601157296) q[6];
rz(2.3249731015322603) q[6];
ry(-2.745403352223705) q[7];
rz(-0.014214597065897125) q[7];
ry(-1.3964634265255818) q[8];
rz(-2.0910775121887664) q[8];
ry(0.0039092784041514506) q[9];
rz(2.3330885163042336) q[9];
ry(-3.141581531908071) q[10];
rz(0.5569533707441213) q[10];
ry(-1.574062234991879) q[11];
rz(-1.473164477176849) q[11];
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
ry(-1.7157662235074698) q[0];
rz(-1.8959258553407352) q[0];
ry(3.0428882317663097) q[1];
rz(-2.043120675303891) q[1];
ry(-2.2940755078590542) q[2];
rz(0.672257713387256) q[2];
ry(3.141003118159658) q[3];
rz(1.5908689152563975) q[3];
ry(-3.136922155505318) q[4];
rz(-2.2376787458147893) q[4];
ry(-0.03322372793221984) q[5];
rz(1.1845822016286016) q[5];
ry(-3.131281793374885) q[6];
rz(2.8542515683474026) q[6];
ry(-2.456782642968458) q[7];
rz(-0.02572024321679667) q[7];
ry(3.139560264379584) q[8];
rz(-2.0915971304352623) q[8];
ry(1.5409811644315061) q[9];
rz(-1.4092028462979558) q[9];
ry(-1.6160766887906624) q[10];
rz(2.189499369362692) q[10];
ry(0.40356644904008393) q[11];
rz(-0.02221998670992636) q[11];
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
ry(-0.9039764410306629) q[0];
rz(2.6973711672986216) q[0];
ry(-1.5970852728105989) q[1];
rz(1.5969803869643517) q[1];
ry(-2.815435866878847) q[2];
rz(1.4182373910969357) q[2];
ry(-1.659513379128609) q[3];
rz(0.007856132771490998) q[3];
ry(1.28017851393254) q[4];
rz(-0.5016873113117416) q[4];
ry(0.07299824164233915) q[5];
rz(-1.142449139322995) q[5];
ry(-0.002531277823029754) q[6];
rz(0.4185737248085202) q[6];
ry(2.875938789925025) q[7];
rz(3.119361163384702) q[7];
ry(1.2782751374270518) q[8];
rz(-0.10940592293214113) q[8];
ry(-3.098936630937626) q[9];
rz(-1.4128132557426005) q[9];
ry(3.123610923909754) q[10];
rz(2.749431769768445) q[10];
ry(-1.4613676188514304) q[11];
rz(-0.0353767746472823) q[11];
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
ry(1.7278782971343558) q[0];
rz(1.592636027070915) q[0];
ry(1.5759483177649998) q[1];
rz(1.4379524255755385) q[1];
ry(0.15886942762654144) q[2];
rz(-0.26338997659069535) q[2];
ry(-1.5701269816266707) q[3];
rz(0.5414282971791002) q[3];
ry(-3.131885338587622) q[4];
rz(-2.8766372392876987) q[4];
ry(3.1223980679722017) q[5];
rz(-0.09996532927629415) q[5];
ry(0.01000299962013429) q[6];
rz(-1.2413202068106317) q[6];
ry(1.8014013706252054) q[7];
rz(-3.1410260180967455) q[7];
ry(0.037945649360683766) q[8];
rz(0.10743926378726432) q[8];
ry(-1.8551743490458463) q[9];
rz(-0.5976012099429644) q[9];
ry(-0.5733506759214402) q[10];
rz(-0.022964543442442142) q[10];
ry(-2.817217051336961) q[11];
rz(-2.7948276903954987) q[11];
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
ry(-1.2059851236935544) q[0];
rz(-0.02140131459752581) q[0];
ry(1.470292710461684) q[1];
rz(-1.4894110534226015) q[1];
ry(-1.569126639528313) q[2];
rz(0.0018983364622329193) q[2];
ry(3.1409172219326305) q[3];
rz(-2.5999993088403097) q[3];
ry(0.17654435562937199) q[4];
rz(-3.115450046751524) q[4];
ry(-2.900080354657537) q[5];
rz(-0.16797601870114942) q[5];
ry(-0.18029871969145633) q[6];
rz(-0.07540777141105294) q[6];
ry(-2.064061925999539) q[7];
rz(3.1400080776270465) q[7];
ry(-3.0484579833174474) q[8];
rz(-0.002610476564350428) q[8];
ry(3.1328267873792472) q[9];
rz(2.550460309853737) q[9];
ry(2.7723724247477937) q[10];
rz(-2.510636161066106) q[10];
ry(-3.005174434899441) q[11];
rz(2.8006451083756625) q[11];
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
ry(1.5188532168028848) q[0];
rz(-1.4627296457247914) q[0];
ry(-1.5723495472080709) q[1];
rz(-2.1957849643929905) q[1];
ry(1.1856014392241834) q[2];
rz(-0.8146315704170624) q[2];
ry(-1.537544397370091) q[3];
rz(-0.16096299127533856) q[3];
ry(0.11817718412462198) q[4];
rz(2.4143663602977696) q[4];
ry(1.5968259460864769) q[5];
rz(0.6492325518917265) q[5];
ry(0.047973629814272556) q[6];
rz(-3.064555037015443) q[6];
ry(-1.5666466479140748) q[7];
rz(2.897385089543822) q[7];
ry(1.5348283213876148) q[8];
rz(0.5323298002655124) q[8];
ry(-0.09266483264519555) q[9];
rz(-0.805377693048449) q[9];
ry(1.1131478518242992) q[10];
rz(-0.3084440714946588) q[10];
ry(-1.5318296811978533) q[11];
rz(-2.3340987032512865) q[11];
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
ry(-0.06734836810372923) q[0];
rz(2.765840991460857) q[0];
ry(-1.1420639772124057) q[1];
rz(-1.6470717049850785) q[1];
ry(1.3410949173842281) q[2];
rz(0.9875893518209412) q[2];
ry(2.380734357844948) q[3];
rz(2.8126810751553193) q[3];
ry(-1.8346760712561139) q[4];
rz(-0.25973427723951076) q[4];
ry(1.6200254191199637) q[5];
rz(0.06058906451859781) q[5];
ry(-2.2534712168377493) q[6];
rz(1.4953308491113102) q[6];
ry(-0.9447219640946711) q[7];
rz(-3.0157688549248234) q[7];
ry(-0.50461777622677) q[8];
rz(-0.06520748656963227) q[8];
ry(-1.4452185075573372) q[9];
rz(0.7737177301436429) q[9];
ry(-1.4061306435154644) q[10];
rz(2.294041283695296) q[10];
ry(-3.038277073695631) q[11];
rz(-2.304014926461608) q[11];
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
ry(-0.00143299942816844) q[0];
rz(0.2635068772766182) q[0];
ry(3.1409188913202164) q[1];
rz(0.44657892576879743) q[1];
ry(-3.1402786198777526) q[2];
rz(2.3225570060852836) q[2];
ry(0.0005883459001010394) q[3];
rz(1.7829858289694571) q[3];
ry(-3.140561711447929) q[4];
rz(-1.6126601108137164) q[4];
ry(3.1412322856018275) q[5];
rz(1.5944093049613972) q[5];
ry(-0.0011980961585766892) q[6];
rz(-3.0700347617471495) q[6];
ry(0.006387008956999728) q[7];
rz(1.5843609173180448) q[7];
ry(3.1402887014347343) q[8];
rz(-1.1606350962762066) q[8];
ry(3.137995460628881) q[9];
rz(2.2161766519588815) q[9];
ry(0.0033789847616301216) q[10];
rz(-0.7276741522880856) q[10];
ry(2.824882249054346) q[11];
rz(2.921849986037034) q[11];
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
ry(-3.063759963067309) q[0];
rz(-2.555694014275974) q[0];
ry(2.3998266425382058) q[1];
rz(0.4705200737964432) q[1];
ry(2.357691599969286) q[2];
rz(1.5168587876601407) q[2];
ry(1.4597557569317028) q[3];
rz(-2.4896122875091793) q[3];
ry(2.246344834806747) q[4];
rz(1.7222532009185922) q[4];
ry(-2.2198965814108993) q[5];
rz(-0.07801561944645574) q[5];
ry(-1.5750944315040423) q[6];
rz(2.2594049057238053) q[6];
ry(-1.7605585399799322) q[7];
rz(-2.0619388468306283) q[7];
ry(-1.8210920250472808) q[8];
rz(-1.6984104903721102) q[8];
ry(0.7811376212605787) q[9];
rz(-1.09779516383721) q[9];
ry(1.5972767238403485) q[10];
rz(-1.9138120200164461) q[10];
ry(-0.08660282635098414) q[11];
rz(-1.1150429767674046) q[11];