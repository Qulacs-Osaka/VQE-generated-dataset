OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.9623445198516318) q[0];
ry(0.8155399240750726) q[1];
cx q[0],q[1];
ry(0.5415891170266756) q[0];
ry(-1.803726418498231) q[1];
cx q[0],q[1];
ry(-1.0004080024985793) q[1];
ry(2.169133218060239) q[2];
cx q[1],q[2];
ry(2.5649452757897206) q[1];
ry(-1.7507236780970992) q[2];
cx q[1],q[2];
ry(2.379722095439394) q[2];
ry(-2.9062885791930886) q[3];
cx q[2],q[3];
ry(-1.3707681594519239) q[2];
ry(-2.9081341169203765) q[3];
cx q[2],q[3];
ry(-0.12933483737883578) q[3];
ry(1.0355145040300013) q[4];
cx q[3],q[4];
ry(-1.9191650122686665) q[3];
ry(-3.0790254253966416) q[4];
cx q[3],q[4];
ry(0.4622305297511657) q[4];
ry(-3.1322428796527384) q[5];
cx q[4],q[5];
ry(-1.4314216068165415) q[4];
ry(-1.869004142840633) q[5];
cx q[4],q[5];
ry(-1.1473751437477748) q[5];
ry(-1.023360434241995) q[6];
cx q[5],q[6];
ry(0.02345783375724898) q[5];
ry(-3.086407253015738) q[6];
cx q[5],q[6];
ry(0.6791347120767983) q[6];
ry(0.43983693093059717) q[7];
cx q[6],q[7];
ry(-1.22205413750378) q[6];
ry(-2.634740102871272) q[7];
cx q[6],q[7];
ry(2.329249847184297) q[7];
ry(-1.449767955155803) q[8];
cx q[7],q[8];
ry(1.8504255080900691) q[7];
ry(1.7036203887447747) q[8];
cx q[7],q[8];
ry(-2.526839602722438) q[8];
ry(-1.5261936286600946) q[9];
cx q[8],q[9];
ry(2.763371763586289) q[8];
ry(3.14104476555254) q[9];
cx q[8],q[9];
ry(-2.3392555220487123) q[9];
ry(3.0681133574837274) q[10];
cx q[9],q[10];
ry(2.326002472727823) q[9];
ry(2.034391319394338) q[10];
cx q[9],q[10];
ry(0.7856733572558714) q[10];
ry(1.4137272646644128) q[11];
cx q[10],q[11];
ry(2.2283834649876795) q[10];
ry(0.6717051955723035) q[11];
cx q[10],q[11];
ry(-2.9814838123118776) q[0];
ry(-1.9508089099773835) q[1];
cx q[0],q[1];
ry(-2.8876054602521513) q[0];
ry(2.79055410707407) q[1];
cx q[0],q[1];
ry(-2.357451452097625) q[1];
ry(-0.9813572197584887) q[2];
cx q[1],q[2];
ry(-2.857374898728063) q[1];
ry(-0.32147396357570424) q[2];
cx q[1],q[2];
ry(-0.9201937994309091) q[2];
ry(-2.922847034336131) q[3];
cx q[2],q[3];
ry(1.8237660437272325) q[2];
ry(-0.36887930887292314) q[3];
cx q[2],q[3];
ry(2.506582801749021) q[3];
ry(-2.7069984633441093) q[4];
cx q[3],q[4];
ry(0.4772432220564991) q[3];
ry(-1.9313891588179857) q[4];
cx q[3],q[4];
ry(-2.5090737623501975) q[4];
ry(0.7475256369929157) q[5];
cx q[4],q[5];
ry(1.9678652014227551) q[4];
ry(-2.7686521574905814) q[5];
cx q[4],q[5];
ry(-1.7496366685899805) q[5];
ry(0.25903191984186025) q[6];
cx q[5],q[6];
ry(3.133354592825582) q[5];
ry(-1.639638099066332) q[6];
cx q[5],q[6];
ry(0.372421118126117) q[6];
ry(1.494198542784878) q[7];
cx q[6],q[7];
ry(1.5293604921477417) q[6];
ry(0.7836420896480879) q[7];
cx q[6],q[7];
ry(-0.8972775347223277) q[7];
ry(-2.777147031820439) q[8];
cx q[7],q[8];
ry(-1.816876044735185) q[7];
ry(0.6342699992307903) q[8];
cx q[7],q[8];
ry(-1.166736391073153) q[8];
ry(2.0263281870696463) q[9];
cx q[8],q[9];
ry(-2.995838744189801) q[8];
ry(0.07286702741376647) q[9];
cx q[8],q[9];
ry(2.1752684455747566) q[9];
ry(-2.484438029512687) q[10];
cx q[9],q[10];
ry(-0.16830749383925792) q[9];
ry(2.1729301596231725) q[10];
cx q[9],q[10];
ry(1.3707623340993011) q[10];
ry(-1.105715470811378) q[11];
cx q[10],q[11];
ry(-1.5918354204804448) q[10];
ry(1.1815118264334075) q[11];
cx q[10],q[11];
ry(-2.943305456000875) q[0];
ry(-2.177924089899265) q[1];
cx q[0],q[1];
ry(-2.49243919937314) q[0];
ry(-0.9235535742763874) q[1];
cx q[0],q[1];
ry(-0.08077661021183147) q[1];
ry(-0.6447483922503967) q[2];
cx q[1],q[2];
ry(1.2739087198155579) q[1];
ry(0.8818684299135205) q[2];
cx q[1],q[2];
ry(1.7185278885567499) q[2];
ry(-2.355476311791844) q[3];
cx q[2],q[3];
ry(-0.3943974759025838) q[2];
ry(-1.0521306199894056) q[3];
cx q[2],q[3];
ry(0.6365389919124845) q[3];
ry(2.1020715821118094) q[4];
cx q[3],q[4];
ry(-2.2756154154452846) q[3];
ry(2.12566713088202) q[4];
cx q[3],q[4];
ry(1.5169536792535347) q[4];
ry(-1.387245330203914) q[5];
cx q[4],q[5];
ry(-1.6918150355302224) q[4];
ry(3.140441029828401) q[5];
cx q[4],q[5];
ry(2.342429118973574) q[5];
ry(-0.4754405422549013) q[6];
cx q[5],q[6];
ry(-0.42706976999576995) q[5];
ry(2.64390178929648) q[6];
cx q[5],q[6];
ry(3.098865886612809) q[6];
ry(-0.2927420427953949) q[7];
cx q[6],q[7];
ry(-0.1655963240983827) q[6];
ry(0.313586251596389) q[7];
cx q[6],q[7];
ry(0.40137141592225806) q[7];
ry(-1.2569360144289243) q[8];
cx q[7],q[8];
ry(0.16248640529447306) q[7];
ry(-0.5570085344929416) q[8];
cx q[7],q[8];
ry(-2.8312885789375133) q[8];
ry(-2.638182291544841) q[9];
cx q[8],q[9];
ry(-1.9866275010614356) q[8];
ry(0.02748237611732233) q[9];
cx q[8],q[9];
ry(0.08917041788962177) q[9];
ry(-0.41404385862446547) q[10];
cx q[9],q[10];
ry(2.068626095094413) q[9];
ry(2.736426696576952) q[10];
cx q[9],q[10];
ry(-1.2413996185833271) q[10];
ry(-2.3609217663799584) q[11];
cx q[10],q[11];
ry(-1.5891304451354835) q[10];
ry(-1.0120910158229393) q[11];
cx q[10],q[11];
ry(-2.430231138092773) q[0];
ry(2.2255358709777493) q[1];
cx q[0],q[1];
ry(1.059107422983599) q[0];
ry(-2.4803786622340316) q[1];
cx q[0],q[1];
ry(-2.1632860236185385) q[1];
ry(1.6304970718629634) q[2];
cx q[1],q[2];
ry(0.5903618038090821) q[1];
ry(2.6210747607206772) q[2];
cx q[1],q[2];
ry(1.430916907405088) q[2];
ry(-2.166036067437539) q[3];
cx q[2],q[3];
ry(2.722719717015832) q[2];
ry(-1.0221892449171317) q[3];
cx q[2],q[3];
ry(0.0641512626556775) q[3];
ry(0.8002685993481214) q[4];
cx q[3],q[4];
ry(-2.3525311636280435) q[3];
ry(-1.2230364157295288) q[4];
cx q[3],q[4];
ry(-2.2877654809751746) q[4];
ry(2.371965490979826) q[5];
cx q[4],q[5];
ry(-2.2576417908336306) q[4];
ry(3.0991373596876253) q[5];
cx q[4],q[5];
ry(2.4319320900833348) q[5];
ry(-0.9804417415044548) q[6];
cx q[5],q[6];
ry(2.9246027932057226) q[5];
ry(2.7083794889879425) q[6];
cx q[5],q[6];
ry(2.5142624586954727) q[6];
ry(2.4870487038859275) q[7];
cx q[6],q[7];
ry(0.8576112219932184) q[6];
ry(1.2860476417668796) q[7];
cx q[6],q[7];
ry(1.2665125924646823) q[7];
ry(-0.41727315133205983) q[8];
cx q[7],q[8];
ry(-2.1759166632030587) q[7];
ry(-2.959156367005984) q[8];
cx q[7],q[8];
ry(1.169251139305576) q[8];
ry(-1.8158381629777451) q[9];
cx q[8],q[9];
ry(-1.9756579332743107) q[8];
ry(-0.06937847324390073) q[9];
cx q[8],q[9];
ry(-1.0452374010595467) q[9];
ry(-0.14991943226949206) q[10];
cx q[9],q[10];
ry(-0.05317386697321718) q[9];
ry(1.3782224688609048) q[10];
cx q[9],q[10];
ry(-1.6407202643617234) q[10];
ry(-0.2075978673358587) q[11];
cx q[10],q[11];
ry(2.9333982186683225) q[10];
ry(-1.09287220251786) q[11];
cx q[10],q[11];
ry(-1.0457478117491865) q[0];
ry(-0.6968665882481186) q[1];
cx q[0],q[1];
ry(1.2018934681553954) q[0];
ry(-1.0753342113366589) q[1];
cx q[0],q[1];
ry(1.7012019756756518) q[1];
ry(1.317150638763439) q[2];
cx q[1],q[2];
ry(0.17962346404772714) q[1];
ry(0.09169808362964685) q[2];
cx q[1],q[2];
ry(-0.16914048554900993) q[2];
ry(-0.4809969167702522) q[3];
cx q[2],q[3];
ry(1.7246028529106414) q[2];
ry(-1.1559441953638538) q[3];
cx q[2],q[3];
ry(-0.18620023600832436) q[3];
ry(0.037170824323955465) q[4];
cx q[3],q[4];
ry(1.347185060270073) q[3];
ry(-1.0144185627597242) q[4];
cx q[3],q[4];
ry(-0.9491189880505875) q[4];
ry(-1.2229842449058184) q[5];
cx q[4],q[5];
ry(3.0701619697435016) q[4];
ry(-0.049438825583335344) q[5];
cx q[4],q[5];
ry(-0.6017690345022916) q[5];
ry(-0.8570285521564784) q[6];
cx q[5],q[6];
ry(1.3310990843213677) q[5];
ry(-0.12194989839187544) q[6];
cx q[5],q[6];
ry(0.3304246911473028) q[6];
ry(-2.1113128975100146) q[7];
cx q[6],q[7];
ry(0.0037650669713820894) q[6];
ry(1.3642336323240256) q[7];
cx q[6],q[7];
ry(-2.35447727330667) q[7];
ry(1.6160589767448836) q[8];
cx q[7],q[8];
ry(0.6360354915633186) q[7];
ry(-3.136320223424552) q[8];
cx q[7],q[8];
ry(2.9748177283236297) q[8];
ry(1.0971040708660853) q[9];
cx q[8],q[9];
ry(1.8548346932749666) q[8];
ry(-0.01953087363695971) q[9];
cx q[8],q[9];
ry(-1.5262849575898612) q[9];
ry(0.07534460302607453) q[10];
cx q[9],q[10];
ry(2.9104339011843012) q[9];
ry(0.21685094664766905) q[10];
cx q[9],q[10];
ry(-0.05034135072715173) q[10];
ry(-0.9317239920789673) q[11];
cx q[10],q[11];
ry(0.2931119937355227) q[10];
ry(3.009326108411335) q[11];
cx q[10],q[11];
ry(-2.8997052022617393) q[0];
ry(-1.2013398698411644) q[1];
cx q[0],q[1];
ry(-3.0875682259183277) q[0];
ry(-2.7094660463928273) q[1];
cx q[0],q[1];
ry(1.149695862536671) q[1];
ry(1.8341248285759644) q[2];
cx q[1],q[2];
ry(1.4468236970692174) q[1];
ry(-2.3532828495334326) q[2];
cx q[1],q[2];
ry(2.804214896198177) q[2];
ry(-2.8322564279768936) q[3];
cx q[2],q[3];
ry(1.9885996671249808) q[2];
ry(-2.303640267225726) q[3];
cx q[2],q[3];
ry(-2.409665081560924) q[3];
ry(2.376044818845151) q[4];
cx q[3],q[4];
ry(-2.012529890267851) q[3];
ry(-2.040909550748335) q[4];
cx q[3],q[4];
ry(-3.0522233312875637) q[4];
ry(-1.792424252608014) q[5];
cx q[4],q[5];
ry(0.12548991531297166) q[4];
ry(-0.012612804585192983) q[5];
cx q[4],q[5];
ry(-1.1291315005234557) q[5];
ry(2.3061461569701893) q[6];
cx q[5],q[6];
ry(2.6125807666043936) q[5];
ry(1.50062293614572) q[6];
cx q[5],q[6];
ry(-0.06962996173802871) q[6];
ry(-1.5594018349865202) q[7];
cx q[6],q[7];
ry(-3.088674959414172) q[6];
ry(-1.1983927873735123) q[7];
cx q[6],q[7];
ry(0.3310055471015377) q[7];
ry(-1.1059719537509134) q[8];
cx q[7],q[8];
ry(-0.4796422235625624) q[7];
ry(-2.908784633550798) q[8];
cx q[7],q[8];
ry(0.24447174679480096) q[8];
ry(1.9834245803456103) q[9];
cx q[8],q[9];
ry(-0.4440204332786891) q[8];
ry(-3.002923228443233) q[9];
cx q[8],q[9];
ry(1.9468256806715571) q[9];
ry(-0.7473832910781103) q[10];
cx q[9],q[10];
ry(2.502034301750183) q[9];
ry(-2.051707852518549) q[10];
cx q[9],q[10];
ry(-0.008716203936116607) q[10];
ry(-1.162832188609003) q[11];
cx q[10],q[11];
ry(-1.0517075589430727) q[10];
ry(1.2271961545562087) q[11];
cx q[10],q[11];
ry(-2.325822116675241) q[0];
ry(-0.09947977463638047) q[1];
cx q[0],q[1];
ry(-3.056226930414179) q[0];
ry(-0.13787328614197) q[1];
cx q[0],q[1];
ry(-0.41007535045613874) q[1];
ry(0.8094967728512807) q[2];
cx q[1],q[2];
ry(-2.1279589111062642) q[1];
ry(1.9388644306916953) q[2];
cx q[1],q[2];
ry(2.8848412269805226) q[2];
ry(1.6591373605878532) q[3];
cx q[2],q[3];
ry(0.8175311385138277) q[2];
ry(-1.6015609498689374) q[3];
cx q[2],q[3];
ry(-2.0812269414964613) q[3];
ry(0.2876565790668666) q[4];
cx q[3],q[4];
ry(0.1521098673596812) q[3];
ry(2.0418024314568246) q[4];
cx q[3],q[4];
ry(-1.8514470859655807) q[4];
ry(2.85460828384468) q[5];
cx q[4],q[5];
ry(0.19270775927954364) q[4];
ry(0.6374320374885158) q[5];
cx q[4],q[5];
ry(1.0650050672154858) q[5];
ry(-2.1479661529408762) q[6];
cx q[5],q[6];
ry(3.043205816026832) q[5];
ry(3.102614429235868) q[6];
cx q[5],q[6];
ry(1.4095240886882328) q[6];
ry(-2.5039801819223273) q[7];
cx q[6],q[7];
ry(-3.08876252932628) q[6];
ry(3.1113352252803064) q[7];
cx q[6],q[7];
ry(-2.0133402563823615) q[7];
ry(0.7881867441308976) q[8];
cx q[7],q[8];
ry(-1.1507199828904415) q[7];
ry(2.5186406369845247) q[8];
cx q[7],q[8];
ry(1.7732246120551836) q[8];
ry(-0.7335926642153104) q[9];
cx q[8],q[9];
ry(-2.367326004325661) q[8];
ry(0.9125747139207698) q[9];
cx q[8],q[9];
ry(0.26465518732963744) q[9];
ry(-2.361529203145959) q[10];
cx q[9],q[10];
ry(-1.7797726773972777) q[9];
ry(1.701236440220208) q[10];
cx q[9],q[10];
ry(-2.1497985717861097) q[10];
ry(2.966094768266814) q[11];
cx q[10],q[11];
ry(-0.7304887793019627) q[10];
ry(-3.0008365557844017) q[11];
cx q[10],q[11];
ry(-0.344168340734566) q[0];
ry(2.7005471444859177) q[1];
cx q[0],q[1];
ry(0.8513086415645614) q[0];
ry(0.9567090044995799) q[1];
cx q[0],q[1];
ry(-0.9475792731282544) q[1];
ry(-0.2703489651955251) q[2];
cx q[1],q[2];
ry(-1.5114452086149557) q[1];
ry(0.24947662459441716) q[2];
cx q[1],q[2];
ry(0.4211945899516669) q[2];
ry(-0.6738719789288483) q[3];
cx q[2],q[3];
ry(0.8590015009621766) q[2];
ry(-0.9869024215709543) q[3];
cx q[2],q[3];
ry(0.5977569477650917) q[3];
ry(-1.3818866021709486) q[4];
cx q[3],q[4];
ry(-0.3668525937256918) q[3];
ry(1.580741561680622) q[4];
cx q[3],q[4];
ry(2.2893074268965248) q[4];
ry(-1.9365590619856385) q[5];
cx q[4],q[5];
ry(0.2345195553175032) q[4];
ry(-0.9749505111116774) q[5];
cx q[4],q[5];
ry(-0.5619850806808334) q[5];
ry(1.6241011165901282) q[6];
cx q[5],q[6];
ry(-2.982578246424972) q[5];
ry(0.06816770544641937) q[6];
cx q[5],q[6];
ry(2.595839601623327) q[6];
ry(-2.842404896064136) q[7];
cx q[6],q[7];
ry(-0.007771863693839725) q[6];
ry(2.6322049854656164) q[7];
cx q[6],q[7];
ry(2.279244318460582) q[7];
ry(2.4117178996836177) q[8];
cx q[7],q[8];
ry(-2.5191784098920214) q[7];
ry(-0.09161692510788891) q[8];
cx q[7],q[8];
ry(-1.6293105729544757) q[8];
ry(2.2421121972444347) q[9];
cx q[8],q[9];
ry(-0.6953157262540639) q[8];
ry(2.386199700266878) q[9];
cx q[8],q[9];
ry(-3.032487802507441) q[9];
ry(-2.281196516714664) q[10];
cx q[9],q[10];
ry(-1.4472517809466598) q[9];
ry(0.14575308836666442) q[10];
cx q[9],q[10];
ry(-1.3410212074015426) q[10];
ry(1.0549558619497628) q[11];
cx q[10],q[11];
ry(0.017078751720122763) q[10];
ry(0.1832400478428469) q[11];
cx q[10],q[11];
ry(-1.9463179636797678) q[0];
ry(0.4612142090713345) q[1];
cx q[0],q[1];
ry(1.1210029303057703) q[0];
ry(1.2935314158175482) q[1];
cx q[0],q[1];
ry(-0.5403198476880133) q[1];
ry(1.044801760729471) q[2];
cx q[1],q[2];
ry(-3.083992551463836) q[1];
ry(0.5686588556828653) q[2];
cx q[1],q[2];
ry(-2.923031545592959) q[2];
ry(-1.5540435852601444) q[3];
cx q[2],q[3];
ry(0.6670992182213755) q[2];
ry(2.2956611598577803) q[3];
cx q[2],q[3];
ry(-0.7891373169777838) q[3];
ry(-0.375605358924644) q[4];
cx q[3],q[4];
ry(0.3301024940057011) q[3];
ry(2.629832892566578) q[4];
cx q[3],q[4];
ry(1.471462414522499) q[4];
ry(1.3533122179638057) q[5];
cx q[4],q[5];
ry(-2.996971722442014) q[4];
ry(-1.1982721109487084) q[5];
cx q[4],q[5];
ry(-1.3099375144753018) q[5];
ry(2.915158825291484) q[6];
cx q[5],q[6];
ry(0.47331139980491965) q[5];
ry(-0.0019099607574332462) q[6];
cx q[5],q[6];
ry(-2.912421690963574) q[6];
ry(-0.8106404003584222) q[7];
cx q[6],q[7];
ry(-3.127985706705123) q[6];
ry(0.4594638737102468) q[7];
cx q[6],q[7];
ry(2.4524843060807453) q[7];
ry(1.6357111235815092) q[8];
cx q[7],q[8];
ry(-1.931385629733621) q[7];
ry(-2.9765234851727285) q[8];
cx q[7],q[8];
ry(-0.6370873475994495) q[8];
ry(-1.4929581336397497) q[9];
cx q[8],q[9];
ry(-0.9922839746779741) q[8];
ry(2.9343605485282476) q[9];
cx q[8],q[9];
ry(-1.6184769430302874) q[9];
ry(1.8789799464408574) q[10];
cx q[9],q[10];
ry(-1.9357584878228025) q[9];
ry(1.1192166483874662) q[10];
cx q[9],q[10];
ry(-2.308523389010869) q[10];
ry(2.615355284995188) q[11];
cx q[10],q[11];
ry(1.2267002507826514) q[10];
ry(-0.18836240979796265) q[11];
cx q[10],q[11];
ry(2.406072845600563) q[0];
ry(-0.3561367479248867) q[1];
cx q[0],q[1];
ry(1.357110186439436) q[0];
ry(-1.2042452546490594) q[1];
cx q[0],q[1];
ry(1.1066075005090559) q[1];
ry(1.378411716215929) q[2];
cx q[1],q[2];
ry(-0.6642535790471218) q[1];
ry(-1.0699234600664704) q[2];
cx q[1],q[2];
ry(-1.2140069310203392) q[2];
ry(2.8896647073610167) q[3];
cx q[2],q[3];
ry(-2.497569767081367) q[2];
ry(-1.5368045772305008) q[3];
cx q[2],q[3];
ry(-2.6624656395709336) q[3];
ry(-2.3111864127610513) q[4];
cx q[3],q[4];
ry(-1.0490726526098069) q[3];
ry(-1.4415876510276702) q[4];
cx q[3],q[4];
ry(-3.1202679176411405) q[4];
ry(1.7547640532198048) q[5];
cx q[4],q[5];
ry(3.136800663635258) q[4];
ry(-2.18526631006553) q[5];
cx q[4],q[5];
ry(-0.45930532445704486) q[5];
ry(-0.8148879761728516) q[6];
cx q[5],q[6];
ry(2.363840575151972) q[5];
ry(-1.0955517073929775) q[6];
cx q[5],q[6];
ry(1.0642302858876906) q[6];
ry(2.209321037335143) q[7];
cx q[6],q[7];
ry(-0.18150286516291378) q[6];
ry(0.0010398196311175467) q[7];
cx q[6],q[7];
ry(-2.2124069997127775) q[7];
ry(-2.282249062258307) q[8];
cx q[7],q[8];
ry(1.2357341508756567) q[7];
ry(-2.833045744981299) q[8];
cx q[7],q[8];
ry(2.0566608445343872) q[8];
ry(-3.105980220449125) q[9];
cx q[8],q[9];
ry(-1.623693054899305) q[8];
ry(-0.13580408845151634) q[9];
cx q[8],q[9];
ry(1.3633749402022106) q[9];
ry(0.8092610141147946) q[10];
cx q[9],q[10];
ry(1.1829305270266233) q[9];
ry(-2.0056112692271664) q[10];
cx q[9],q[10];
ry(-1.2483902831932145) q[10];
ry(-1.1334754906443065) q[11];
cx q[10],q[11];
ry(-1.1973002907361385) q[10];
ry(-1.1119696578186957) q[11];
cx q[10],q[11];
ry(2.4950794437548387) q[0];
ry(-0.28740608494751496) q[1];
cx q[0],q[1];
ry(0.41970176999861314) q[0];
ry(-1.3339167719381437) q[1];
cx q[0],q[1];
ry(2.1650477040070197) q[1];
ry(-2.4309401623845375) q[2];
cx q[1],q[2];
ry(-0.457398746179724) q[1];
ry(0.5728409988230123) q[2];
cx q[1],q[2];
ry(2.591617174737811) q[2];
ry(2.8251448364625413) q[3];
cx q[2],q[3];
ry(-1.8774629699222518) q[2];
ry(0.6324813774313691) q[3];
cx q[2],q[3];
ry(2.345506377750432) q[3];
ry(-2.6346679735638747) q[4];
cx q[3],q[4];
ry(-1.263150986522611) q[3];
ry(-1.3387950278437637) q[4];
cx q[3],q[4];
ry(-1.2572007361899602) q[4];
ry(1.5651520204321459) q[5];
cx q[4],q[5];
ry(-1.4938307547683671) q[4];
ry(-2.699633807169573) q[5];
cx q[4],q[5];
ry(-1.5221757805416174) q[5];
ry(1.31063461561682) q[6];
cx q[5],q[6];
ry(3.1343674224477933) q[5];
ry(2.0892371583834937) q[6];
cx q[5],q[6];
ry(2.9827747545710146) q[6];
ry(-3.0673297666169717) q[7];
cx q[6],q[7];
ry(3.1160722618538763) q[6];
ry(2.979196871531466) q[7];
cx q[6],q[7];
ry(1.8771703704132539) q[7];
ry(0.28390978265655953) q[8];
cx q[7],q[8];
ry(1.9200402032872077) q[7];
ry(0.46170932270943593) q[8];
cx q[7],q[8];
ry(-1.5862252168304947) q[8];
ry(2.6582713178291493) q[9];
cx q[8],q[9];
ry(1.3768268001572004) q[8];
ry(-2.9928398769758493) q[9];
cx q[8],q[9];
ry(2.4008871755220547) q[9];
ry(2.3881948263024593) q[10];
cx q[9],q[10];
ry(-0.5371086422362702) q[9];
ry(-0.06969176260210741) q[10];
cx q[9],q[10];
ry(1.4969003297509094) q[10];
ry(2.0741526275026434) q[11];
cx q[10],q[11];
ry(-1.328511507986815) q[10];
ry(2.0558891138396147) q[11];
cx q[10],q[11];
ry(1.9047454268392356) q[0];
ry(2.7441845695646863) q[1];
cx q[0],q[1];
ry(-1.9100847936076186) q[0];
ry(-2.5627439778111407) q[1];
cx q[0],q[1];
ry(-1.7722892740430476) q[1];
ry(2.3571996567850033) q[2];
cx q[1],q[2];
ry(0.5951280116577489) q[1];
ry(2.984662913068605) q[2];
cx q[1],q[2];
ry(-1.5804146103653833) q[2];
ry(-1.0613428887631962) q[3];
cx q[2],q[3];
ry(-1.6965662339957897) q[2];
ry(1.636205127673238) q[3];
cx q[2],q[3];
ry(0.6487718801295337) q[3];
ry(1.8656983473231215) q[4];
cx q[3],q[4];
ry(0.038406052930376265) q[3];
ry(2.96607328075484) q[4];
cx q[3],q[4];
ry(1.310981788547224) q[4];
ry(-1.4782369547910195) q[5];
cx q[4],q[5];
ry(-1.5726612888707847) q[4];
ry(1.5855548164388331) q[5];
cx q[4],q[5];
ry(-0.12072523636475217) q[5];
ry(-0.38891600150174266) q[6];
cx q[5],q[6];
ry(-2.945969248957389) q[5];
ry(1.0262102895064857) q[6];
cx q[5],q[6];
ry(-0.7274592217816093) q[6];
ry(-0.8479475624326911) q[7];
cx q[6],q[7];
ry(0.1224505237517759) q[6];
ry(1.249892346864697) q[7];
cx q[6],q[7];
ry(-0.8003242937186918) q[7];
ry(3.0371463521456143) q[8];
cx q[7],q[8];
ry(1.7754165618150337) q[7];
ry(1.8383277283511141) q[8];
cx q[7],q[8];
ry(-3.0265192386063697) q[8];
ry(-1.6833291414599822) q[9];
cx q[8],q[9];
ry(-2.086717078264453) q[8];
ry(-3.125259942753195) q[9];
cx q[8],q[9];
ry(0.3188462082629284) q[9];
ry(0.6353527110391637) q[10];
cx q[9],q[10];
ry(-1.4916218305905973) q[9];
ry(-0.607854242387118) q[10];
cx q[9],q[10];
ry(1.0965365156376066) q[10];
ry(2.835177387128712) q[11];
cx q[10],q[11];
ry(-0.46254657223097606) q[10];
ry(2.421833151212723) q[11];
cx q[10],q[11];
ry(1.142374268955761) q[0];
ry(-2.5987081726616803) q[1];
cx q[0],q[1];
ry(1.266630137154019) q[0];
ry(-2.3505751752631214) q[1];
cx q[0],q[1];
ry(-0.956273309212918) q[1];
ry(0.16871065370654748) q[2];
cx q[1],q[2];
ry(1.0796535461896974) q[1];
ry(1.3563865919534868) q[2];
cx q[1],q[2];
ry(2.2482944111198275) q[2];
ry(1.5503192659876674) q[3];
cx q[2],q[3];
ry(-1.714674584242853) q[2];
ry(-0.6244033953930954) q[3];
cx q[2],q[3];
ry(2.887892822235062) q[3];
ry(-1.7997453657085494) q[4];
cx q[3],q[4];
ry(0.5946293987985437) q[3];
ry(-0.502476561166912) q[4];
cx q[3],q[4];
ry(2.9401599867301247) q[4];
ry(-0.002774804096546544) q[5];
cx q[4],q[5];
ry(-3.1203814252461073) q[4];
ry(3.10095367164601) q[5];
cx q[4],q[5];
ry(-2.8027318236447702) q[5];
ry(-1.2356098056344331) q[6];
cx q[5],q[6];
ry(3.127050820962955) q[5];
ry(-2.916282413106983) q[6];
cx q[5],q[6];
ry(-1.2217999484896696) q[6];
ry(3.0467955455893043) q[7];
cx q[6],q[7];
ry(0.23599953288983322) q[6];
ry(1.524948333997407) q[7];
cx q[6],q[7];
ry(2.8155854365111956) q[7];
ry(-0.24183339520063193) q[8];
cx q[7],q[8];
ry(0.7785992751265036) q[7];
ry(-1.4241161241377673) q[8];
cx q[7],q[8];
ry(1.0049530787351388) q[8];
ry(-1.289145285230405) q[9];
cx q[8],q[9];
ry(1.1500038538585207) q[8];
ry(-2.0979880894523375) q[9];
cx q[8],q[9];
ry(2.7403419280490904) q[9];
ry(-1.522964530313854) q[10];
cx q[9],q[10];
ry(-1.0176373931735287) q[9];
ry(-3.1324214254142135) q[10];
cx q[9],q[10];
ry(2.377783396074479) q[10];
ry(1.2621889864644213) q[11];
cx q[10],q[11];
ry(0.19189707842990167) q[10];
ry(1.0720891958943568) q[11];
cx q[10],q[11];
ry(-1.9964920071999588) q[0];
ry(0.31744190849313414) q[1];
cx q[0],q[1];
ry(-1.8335287353468124) q[0];
ry(1.6782737272458086) q[1];
cx q[0],q[1];
ry(-1.5982208597116454) q[1];
ry(-3.0633250265944083) q[2];
cx q[1],q[2];
ry(-0.5119553617679351) q[1];
ry(1.1778797591736128) q[2];
cx q[1],q[2];
ry(2.034380673988343) q[2];
ry(2.4119325825925633) q[3];
cx q[2],q[3];
ry(0.6129428866212351) q[2];
ry(2.9206773209224193) q[3];
cx q[2],q[3];
ry(1.7366843792765714) q[3];
ry(-2.3084927098974566) q[4];
cx q[3],q[4];
ry(-1.701086464433035) q[3];
ry(1.0763835223337423) q[4];
cx q[3],q[4];
ry(2.984781713882028) q[4];
ry(1.212952407366621) q[5];
cx q[4],q[5];
ry(-3.007132274883875) q[4];
ry(0.01697526547956585) q[5];
cx q[4],q[5];
ry(-1.2985589743629058) q[5];
ry(1.5708947390809265) q[6];
cx q[5],q[6];
ry(0.24844733487777848) q[5];
ry(-0.02623549731578634) q[6];
cx q[5],q[6];
ry(1.6142186720695864) q[6];
ry(-1.5914000198602987) q[7];
cx q[6],q[7];
ry(0.07838903242241207) q[6];
ry(1.8354991212369516) q[7];
cx q[6],q[7];
ry(2.1944785180297006) q[7];
ry(3.135704090007114) q[8];
cx q[7],q[8];
ry(-1.5936416990406794) q[7];
ry(1.9549900788645878) q[8];
cx q[7],q[8];
ry(1.5407116422538243) q[8];
ry(1.680868887371843) q[9];
cx q[8],q[9];
ry(1.5950631990525015) q[8];
ry(0.46190361803820906) q[9];
cx q[8],q[9];
ry(1.903207994471856) q[9];
ry(-2.419325562208307) q[10];
cx q[9],q[10];
ry(-1.7231458130431612) q[9];
ry(-2.8736065717031667) q[10];
cx q[9],q[10];
ry(-2.948850097664319) q[10];
ry(-0.34233753494968444) q[11];
cx q[10],q[11];
ry(1.648123095514114) q[10];
ry(-0.5920636380300515) q[11];
cx q[10],q[11];
ry(-1.6827022849025384) q[0];
ry(-2.658338249389478) q[1];
cx q[0],q[1];
ry(2.722137844183543) q[0];
ry(0.7529526861941722) q[1];
cx q[0],q[1];
ry(1.1426140051784113) q[1];
ry(-1.9589643705457087) q[2];
cx q[1],q[2];
ry(-2.9050173664924515) q[1];
ry(2.876985579465027) q[2];
cx q[1],q[2];
ry(-0.6598235068605165) q[2];
ry(-0.5248243508368737) q[3];
cx q[2],q[3];
ry(-0.3036211239580764) q[2];
ry(-1.7729045654857611) q[3];
cx q[2],q[3];
ry(0.5627928232235767) q[3];
ry(1.2641541943600387) q[4];
cx q[3],q[4];
ry(-2.4948690903254067) q[3];
ry(0.35399015160831393) q[4];
cx q[3],q[4];
ry(2.466562085265344) q[4];
ry(0.5443211834542696) q[5];
cx q[4],q[5];
ry(-0.4417696560580552) q[4];
ry(0.0937012502809542) q[5];
cx q[4],q[5];
ry(1.9494124208813572) q[5];
ry(1.4681119152218618) q[6];
cx q[5],q[6];
ry(3.1092467969201247) q[5];
ry(2.4687115666882558) q[6];
cx q[5],q[6];
ry(2.102536633951101) q[6];
ry(0.2118471356598103) q[7];
cx q[6],q[7];
ry(3.119791076993024) q[6];
ry(3.074169798136788) q[7];
cx q[6],q[7];
ry(-1.5453330481850824) q[7];
ry(-1.8653289064479193) q[8];
cx q[7],q[8];
ry(2.7311538778845534) q[7];
ry(1.97260267113557) q[8];
cx q[7],q[8];
ry(1.8673840497005356) q[8];
ry(1.593536411201196) q[9];
cx q[8],q[9];
ry(-2.3666113861104137) q[8];
ry(1.414271520114967) q[9];
cx q[8],q[9];
ry(-1.2145520062012167) q[9];
ry(-1.791415795674311) q[10];
cx q[9],q[10];
ry(-1.359642109881907) q[9];
ry(-0.7671217876224234) q[10];
cx q[9],q[10];
ry(0.8864039307215869) q[10];
ry(-2.984405652253063) q[11];
cx q[10],q[11];
ry(1.987277090652559) q[10];
ry(-1.9146000682959219) q[11];
cx q[10],q[11];
ry(-2.073192479993088) q[0];
ry(-0.12555560943518776) q[1];
cx q[0],q[1];
ry(1.7523312056187237) q[0];
ry(0.5070516485465505) q[1];
cx q[0],q[1];
ry(-1.3605790448757231) q[1];
ry(-2.535765858097129) q[2];
cx q[1],q[2];
ry(-1.9216924706753655) q[1];
ry(1.7216647016256008) q[2];
cx q[1],q[2];
ry(-2.174821867320695) q[2];
ry(2.5694341954923448) q[3];
cx q[2],q[3];
ry(1.219590628404287) q[2];
ry(1.7883826637521294) q[3];
cx q[2],q[3];
ry(2.922390732812298) q[3];
ry(-0.714222959042534) q[4];
cx q[3],q[4];
ry(-1.661730247919216) q[3];
ry(3.085315161329801) q[4];
cx q[3],q[4];
ry(-3.1157852703828253) q[4];
ry(-1.6690150572065765) q[5];
cx q[4],q[5];
ry(0.22944736594518986) q[4];
ry(0.08712764741318302) q[5];
cx q[4],q[5];
ry(-1.4451715778296288) q[5];
ry(1.881644605699453) q[6];
cx q[5],q[6];
ry(0.12716207736295537) q[5];
ry(-1.3642474578300918) q[6];
cx q[5],q[6];
ry(3.1040992136948313) q[6];
ry(2.4039152877894723) q[7];
cx q[6],q[7];
ry(0.3114670599906464) q[6];
ry(-0.006774159513653011) q[7];
cx q[6],q[7];
ry(-1.9360000302497538) q[7];
ry(-1.2148243213242802) q[8];
cx q[7],q[8];
ry(3.12959828801815) q[7];
ry(1.2582291855067194) q[8];
cx q[7],q[8];
ry(-1.9833100705481967) q[8];
ry(-1.4622904731085846) q[9];
cx q[8],q[9];
ry(2.2845442828256775) q[8];
ry(-1.839140750448239) q[9];
cx q[8],q[9];
ry(-2.4778163965806135) q[9];
ry(2.247395140841913) q[10];
cx q[9],q[10];
ry(-0.035637115294684385) q[9];
ry(2.858316251539634) q[10];
cx q[9],q[10];
ry(-1.0541712252589583) q[10];
ry(2.7924959450972366) q[11];
cx q[10],q[11];
ry(-1.9390426731234296) q[10];
ry(2.4495459003462057) q[11];
cx q[10],q[11];
ry(1.6166336292891512) q[0];
ry(-2.126850359954126) q[1];
cx q[0],q[1];
ry(2.5982442497540563) q[0];
ry(2.839365849032555) q[1];
cx q[0],q[1];
ry(0.9535338449316864) q[1];
ry(1.9332503597709145) q[2];
cx q[1],q[2];
ry(-1.2510873998469387) q[1];
ry(2.712668229001204) q[2];
cx q[1],q[2];
ry(2.384453059023395) q[2];
ry(2.979146532568315) q[3];
cx q[2],q[3];
ry(3.0754622856984666) q[2];
ry(-2.533227509677265) q[3];
cx q[2],q[3];
ry(1.3745961904018769) q[3];
ry(2.895440654235959) q[4];
cx q[3],q[4];
ry(0.8557629015643187) q[3];
ry(-1.5800974751862267) q[4];
cx q[3],q[4];
ry(-1.7160299452505747) q[4];
ry(1.8549158100976222) q[5];
cx q[4],q[5];
ry(-0.15891339611791722) q[4];
ry(3.099948801603396) q[5];
cx q[4],q[5];
ry(-1.868288800625998) q[5];
ry(-1.175158862743804) q[6];
cx q[5],q[6];
ry(0.013870177081297896) q[5];
ry(-1.621729053443126) q[6];
cx q[5],q[6];
ry(2.854475819751983) q[6];
ry(1.7671457965563173) q[7];
cx q[6],q[7];
ry(0.638346913870496) q[6];
ry(-0.32579139025552983) q[7];
cx q[6],q[7];
ry(0.8389246601207091) q[7];
ry(0.022779484710068784) q[8];
cx q[7],q[8];
ry(3.0386365087469938) q[7];
ry(2.6084676167748757) q[8];
cx q[7],q[8];
ry(-2.7450331167658653) q[8];
ry(-0.3272216167838806) q[9];
cx q[8],q[9];
ry(2.4962168459041094) q[8];
ry(-2.6004776542354406) q[9];
cx q[8],q[9];
ry(2.6233686539606826) q[9];
ry(-2.484269667757889) q[10];
cx q[9],q[10];
ry(2.581618021293952) q[9];
ry(-2.2835286420639074) q[10];
cx q[9],q[10];
ry(0.4800470689726186) q[10];
ry(2.8888247591633354) q[11];
cx q[10],q[11];
ry(2.962447557239852) q[10];
ry(0.9645706737110638) q[11];
cx q[10],q[11];
ry(2.423493185203537) q[0];
ry(2.1657548505242827) q[1];
cx q[0],q[1];
ry(-1.7498772576543624) q[0];
ry(-0.991593560819406) q[1];
cx q[0],q[1];
ry(0.822135578139145) q[1];
ry(0.23093426619573254) q[2];
cx q[1],q[2];
ry(0.26006608539313536) q[1];
ry(-1.6717252102698321) q[2];
cx q[1],q[2];
ry(0.9420894355914579) q[2];
ry(-2.9561160651314804) q[3];
cx q[2],q[3];
ry(-2.5378754674732424) q[2];
ry(-0.017836156621029886) q[3];
cx q[2],q[3];
ry(0.355486130297904) q[3];
ry(2.55654304311559) q[4];
cx q[3],q[4];
ry(-0.7985905937249651) q[3];
ry(-0.5436379980279444) q[4];
cx q[3],q[4];
ry(2.6134178852363674) q[4];
ry(-1.5295485846215664) q[5];
cx q[4],q[5];
ry(0.4074275158012002) q[4];
ry(-0.06671403085083405) q[5];
cx q[4],q[5];
ry(2.37001899438084) q[5];
ry(-2.7171485948577168) q[6];
cx q[5],q[6];
ry(0.02361734379152125) q[5];
ry(-3.1332284391590894) q[6];
cx q[5],q[6];
ry(1.3093281410722428) q[6];
ry(2.8432609567055613) q[7];
cx q[6],q[7];
ry(2.1432110388519128) q[6];
ry(-1.532191749856386) q[7];
cx q[6],q[7];
ry(-2.7537172055939534) q[7];
ry(-2.251496808548311) q[8];
cx q[7],q[8];
ry(0.040998441328986786) q[7];
ry(-3.086310054820802) q[8];
cx q[7],q[8];
ry(1.9338428060155683) q[8];
ry(-2.9423011521688944) q[9];
cx q[8],q[9];
ry(-1.8679147418345325) q[8];
ry(-1.515759481420108) q[9];
cx q[8],q[9];
ry(0.21816940142083147) q[9];
ry(-1.856442099524089) q[10];
cx q[9],q[10];
ry(1.617372529336946) q[9];
ry(-0.1150596214012749) q[10];
cx q[9],q[10];
ry(-1.350105578201112) q[10];
ry(1.9153556499993827) q[11];
cx q[10],q[11];
ry(1.6244800292674102) q[10];
ry(0.1218635211320551) q[11];
cx q[10],q[11];
ry(2.8114468626564957) q[0];
ry(2.8234930698227676) q[1];
cx q[0],q[1];
ry(1.4698458360766977) q[0];
ry(2.23612327347497) q[1];
cx q[0],q[1];
ry(-1.8939661262249192) q[1];
ry(-0.12683865213777545) q[2];
cx q[1],q[2];
ry(1.8420491334690619) q[1];
ry(1.1807776498662514) q[2];
cx q[1],q[2];
ry(-0.3454635695807804) q[2];
ry(2.384415914336252) q[3];
cx q[2],q[3];
ry(0.3437477911673446) q[2];
ry(1.9855570432101959) q[3];
cx q[2],q[3];
ry(1.3526305963188652) q[3];
ry(-1.9017824992159085) q[4];
cx q[3],q[4];
ry(-2.4109806202860904) q[3];
ry(2.745056908627866) q[4];
cx q[3],q[4];
ry(2.5373474466874186) q[4];
ry(2.5769879400468167) q[5];
cx q[4],q[5];
ry(0.3128827924853095) q[4];
ry(-3.029733230517705) q[5];
cx q[4],q[5];
ry(1.2263653140555686) q[5];
ry(-0.08412738544942193) q[6];
cx q[5],q[6];
ry(-3.117237971579192) q[5];
ry(-0.05405886226762543) q[6];
cx q[5],q[6];
ry(2.1754451313600565) q[6];
ry(-1.527110868090424) q[7];
cx q[6],q[7];
ry(-0.7601135105179555) q[6];
ry(0.042526757019526684) q[7];
cx q[6],q[7];
ry(-1.3167793335686877) q[7];
ry(1.1440494201123297) q[8];
cx q[7],q[8];
ry(3.117058151540375) q[7];
ry(-0.10772906619421851) q[8];
cx q[7],q[8];
ry(2.2097226607391796) q[8];
ry(0.9524018493812543) q[9];
cx q[8],q[9];
ry(-1.6006071945858587) q[8];
ry(-0.8954302234189374) q[9];
cx q[8],q[9];
ry(-0.798734900192084) q[9];
ry(-0.056758471453607136) q[10];
cx q[9],q[10];
ry(-1.3022559455528064) q[9];
ry(-2.5937500508239344) q[10];
cx q[9],q[10];
ry(2.4822066337608484) q[10];
ry(1.3051970249762883) q[11];
cx q[10],q[11];
ry(-2.195442548335385) q[10];
ry(2.252665165564276) q[11];
cx q[10],q[11];
ry(-1.8912579616739027) q[0];
ry(-1.9694172026402175) q[1];
cx q[0],q[1];
ry(3.0909965564798996) q[0];
ry(2.1770968253445098) q[1];
cx q[0],q[1];
ry(1.6658372451924326) q[1];
ry(-0.05793833315877616) q[2];
cx q[1],q[2];
ry(-2.005551860800767) q[1];
ry(-2.450588549488077) q[2];
cx q[1],q[2];
ry(-1.3105672471412821) q[2];
ry(-1.6397947376681543) q[3];
cx q[2],q[3];
ry(3.0265773627646335) q[2];
ry(0.6545502671789398) q[3];
cx q[2],q[3];
ry(-1.68304974132179) q[3];
ry(0.452892166053779) q[4];
cx q[3],q[4];
ry(-1.641970218673671) q[3];
ry(2.2167476147734977) q[4];
cx q[3],q[4];
ry(0.3226492698499337) q[4];
ry(-2.8719303861038656) q[5];
cx q[4],q[5];
ry(-3.135095816093971) q[4];
ry(-3.11368025140692) q[5];
cx q[4],q[5];
ry(-2.9908060558875733) q[5];
ry(-1.992998381942685) q[6];
cx q[5],q[6];
ry(1.6490069793061686) q[5];
ry(-1.512384724280318) q[6];
cx q[5],q[6];
ry(-0.6813006404734674) q[6];
ry(-2.8274203446976265) q[7];
cx q[6],q[7];
ry(-1.5688148874340113) q[6];
ry(1.589416607379316) q[7];
cx q[6],q[7];
ry(-2.117422280815374) q[7];
ry(-0.7012413857870649) q[8];
cx q[7],q[8];
ry(0.00047652449386693696) q[7];
ry(0.004572383727826278) q[8];
cx q[7],q[8];
ry(0.9668761207602855) q[8];
ry(2.818263162095971) q[9];
cx q[8],q[9];
ry(2.484067720162051) q[8];
ry(3.137236292989113) q[9];
cx q[8],q[9];
ry(-0.513005667685248) q[9];
ry(2.6396099550027894) q[10];
cx q[9],q[10];
ry(1.487002118889359) q[9];
ry(-1.701330356701222) q[10];
cx q[9],q[10];
ry(0.14233395677881155) q[10];
ry(-0.2166468417602312) q[11];
cx q[10],q[11];
ry(0.192499869813599) q[10];
ry(-1.345805111908576) q[11];
cx q[10],q[11];
ry(0.5586627517123759) q[0];
ry(1.8707562367327037) q[1];
cx q[0],q[1];
ry(1.2990215041785382) q[0];
ry(-0.7021126134059347) q[1];
cx q[0],q[1];
ry(2.5039909333640047) q[1];
ry(-2.257365792334347) q[2];
cx q[1],q[2];
ry(1.7416885953274845) q[1];
ry(1.9921678371194487) q[2];
cx q[1],q[2];
ry(1.3561073939233124) q[2];
ry(-2.4487641493887056) q[3];
cx q[2],q[3];
ry(-3.0964183972908965) q[2];
ry(2.79066621454585) q[3];
cx q[2],q[3];
ry(0.89044057302214) q[3];
ry(2.0112290997902718) q[4];
cx q[3],q[4];
ry(0.5481758478703309) q[3];
ry(1.5009104235985558) q[4];
cx q[3],q[4];
ry(1.683643657458366) q[4];
ry(1.713793583357136) q[5];
cx q[4],q[5];
ry(0.0008132934213725222) q[4];
ry(3.1353612735568093) q[5];
cx q[4],q[5];
ry(2.8842142842499308) q[5];
ry(-0.009004262517217266) q[6];
cx q[5],q[6];
ry(-1.5800900893108232) q[5];
ry(0.006959120563085896) q[6];
cx q[5],q[6];
ry(-3.011887187928892) q[6];
ry(-1.0421623375371167) q[7];
cx q[6],q[7];
ry(-1.5764135198075973) q[6];
ry(1.7128764676053903) q[7];
cx q[6],q[7];
ry(-0.004297784594124998) q[7];
ry(3.045221867001557) q[8];
cx q[7],q[8];
ry(3.137769100982755) q[7];
ry(-0.0818761262619817) q[8];
cx q[7],q[8];
ry(-1.6058432072942073) q[8];
ry(2.910042772683212) q[9];
cx q[8],q[9];
ry(2.082228847390413) q[8];
ry(2.519414950075659) q[9];
cx q[8],q[9];
ry(0.2294917049987478) q[9];
ry(1.8883245624412677) q[10];
cx q[9],q[10];
ry(-1.6433240139755876) q[9];
ry(-0.23750702799648593) q[10];
cx q[9],q[10];
ry(-2.589525896652202) q[10];
ry(-2.3227270471914108) q[11];
cx q[10],q[11];
ry(1.8132877025629746) q[10];
ry(2.4692436292044215) q[11];
cx q[10],q[11];
ry(-0.25485883922969776) q[0];
ry(0.2969548064092882) q[1];
cx q[0],q[1];
ry(1.229863457393794) q[0];
ry(0.5488532774016983) q[1];
cx q[0],q[1];
ry(-0.6443006661850301) q[1];
ry(-0.04121292725708368) q[2];
cx q[1],q[2];
ry(0.751787036190473) q[1];
ry(-2.5357988653198165) q[2];
cx q[1],q[2];
ry(2.7611504702547776) q[2];
ry(-0.5807933631576763) q[3];
cx q[2],q[3];
ry(-0.08045523155947176) q[2];
ry(0.2702214647757038) q[3];
cx q[2],q[3];
ry(2.274788143672561) q[3];
ry(1.720860489465794) q[4];
cx q[3],q[4];
ry(-1.9518634242159383) q[3];
ry(-1.454350165558724) q[4];
cx q[3],q[4];
ry(-1.2280274419273607) q[4];
ry(-0.1087549983093768) q[5];
cx q[4],q[5];
ry(3.141219251556463) q[4];
ry(-3.1413182263855246) q[5];
cx q[4],q[5];
ry(2.914861102270322) q[5];
ry(-2.446182705258077) q[6];
cx q[5],q[6];
ry(1.6356474491912796) q[5];
ry(-1.5616698739128534) q[6];
cx q[5],q[6];
ry(-2.124257521162277) q[6];
ry(1.411913591432305) q[7];
cx q[6],q[7];
ry(0.14203839961315637) q[6];
ry(1.5714528884863403) q[7];
cx q[6],q[7];
ry(0.16986478254905923) q[7];
ry(1.4709438442844902) q[8];
cx q[7],q[8];
ry(-1.57220300581436) q[7];
ry(-0.6734706068030203) q[8];
cx q[7],q[8];
ry(-1.5814576025827731) q[8];
ry(-1.6285333027635465) q[9];
cx q[8],q[9];
ry(1.5641627865171133) q[8];
ry(-2.910350643058014) q[9];
cx q[8],q[9];
ry(1.5693829035073144) q[9];
ry(-1.4087131524425287) q[10];
cx q[9],q[10];
ry(1.570326941546639) q[9];
ry(1.5155352050197415) q[10];
cx q[9],q[10];
ry(-1.5698858903392334) q[10];
ry(-1.0950533359689194) q[11];
cx q[10],q[11];
ry(1.5691478104767322) q[10];
ry(-1.535840849287542) q[11];
cx q[10],q[11];
ry(0.7807060663843489) q[0];
ry(0.32471929098464963) q[1];
ry(-0.29682681830464736) q[2];
ry(-1.138823938658903) q[3];
ry(2.7868471761860407) q[4];
ry(-0.7985253368399022) q[5];
ry(2.4045338031378747) q[6];
ry(-2.323437281347291) q[7];
ry(-2.326127718657116) q[8];
ry(0.8242856417980091) q[9];
ry(-2.3170080766402967) q[10];
ry(-2.310332797655845) q[11];