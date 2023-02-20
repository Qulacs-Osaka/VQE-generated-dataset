OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-2.9811310104143276) q[0];
rz(3.0911742591735702) q[0];
ry(-1.9532386041003489) q[1];
rz(1.3314295116727868) q[1];
ry(-1.8536679629754942) q[2];
rz(3.038667915316416) q[2];
ry(3.140194806361636) q[3];
rz(1.3055780421673782) q[3];
ry(-2.6319215147567996) q[4];
rz(1.854718768817485) q[4];
ry(3.1195329151154665) q[5];
rz(-2.656602737573561) q[5];
ry(-3.14090880131909) q[6];
rz(-2.2368656843333152) q[6];
ry(-0.005399706487188446) q[7];
rz(2.7777297706100663) q[7];
ry(-3.111179484113106) q[8];
rz(3.0476287159426763) q[8];
ry(-0.8573462585792093) q[9];
rz(2.458601722153632) q[9];
ry(-2.801031375286764) q[10];
rz(1.699543677449637) q[10];
ry(3.1310160950236146) q[11];
rz(3.0584676414092073) q[11];
ry(2.360994120996983) q[12];
rz(-1.6653861548632234) q[12];
ry(-0.13467263286099218) q[13];
rz(0.08947281008551755) q[13];
ry(-0.6170128378004794) q[14];
rz(0.5470043820604318) q[14];
ry(1.2103015945060616) q[15];
rz(1.629021856486362) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.036244476771871696) q[0];
rz(-2.3594426623905) q[0];
ry(2.4101699703524666) q[1];
rz(1.116138518401051) q[1];
ry(-2.4588899914618136) q[2];
rz(2.7510532415977815) q[2];
ry(3.1408822210699845) q[3];
rz(2.319704395930378) q[3];
ry(-0.4599687396453822) q[4];
rz(-1.0946712558613934) q[4];
ry(3.130050953028313) q[5];
rz(0.882187829690488) q[5];
ry(-0.0055057617042409725) q[6];
rz(-2.386063803835191) q[6];
ry(0.013005016128955481) q[7];
rz(-0.0333115606975928) q[7];
ry(0.025299014271458337) q[8];
rz(3.1269734059017673) q[8];
ry(0.8671007696078218) q[9];
rz(0.9989008165725037) q[9];
ry(-2.370152359350613) q[10];
rz(0.2012999232952586) q[10];
ry(-0.5810554888574311) q[11];
rz(-2.8521926415280316) q[11];
ry(2.4168827112655964) q[12];
rz(-2.95314885988016) q[12];
ry(1.5898195397220007) q[13];
rz(-2.5376879121574256) q[13];
ry(1.8025546439135667) q[14];
rz(2.71999117422976) q[14];
ry(0.6866762939543563) q[15];
rz(-2.5001943526316466) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.031679223756806735) q[0];
rz(-0.17179362191724312) q[0];
ry(2.552135099794557) q[1];
rz(2.8188396676301037) q[1];
ry(-0.7612424420831045) q[2];
rz(-1.345777884718354) q[2];
ry(-0.0010361092653263298) q[3];
rz(-0.6203677817746812) q[3];
ry(0.01775477748073783) q[4];
rz(1.6207135087619653) q[4];
ry(-1.6130421454277108) q[5];
rz(-1.6680745732682098) q[5];
ry(-0.018900092806065416) q[6];
rz(1.8532593833787727) q[6];
ry(0.0041743863402237125) q[7];
rz(0.1870709997863676) q[7];
ry(0.8362022877929947) q[8];
rz(0.6586317611399528) q[8];
ry(3.1139388721884442) q[9];
rz(-2.64485666571914) q[9];
ry(-0.3495646699810022) q[10];
rz(-2.708799754988925) q[10];
ry(3.1315027162500124) q[11];
rz(-0.6553972127397846) q[11];
ry(-2.701361187100789) q[12];
rz(1.9300442152695978) q[12];
ry(-0.34139549888426934) q[13];
rz(-1.9128404530664074) q[13];
ry(1.1309794958007457) q[14];
rz(-0.4786484453972957) q[14];
ry(-1.2234422994270995) q[15];
rz(0.9625126918406882) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.01292915451339649) q[0];
rz(-1.7710766984160033) q[0];
ry(-2.303788213796129) q[1];
rz(-0.034471850526044186) q[1];
ry(0.2147682509052835) q[2];
rz(-2.8813750517712586) q[2];
ry(7.195618157029107e-05) q[3];
rz(2.2160625920722747) q[3];
ry(0.0066903197812597974) q[4];
rz(-3.0747209235527024) q[4];
ry(2.9024174460188563) q[5];
rz(-0.8457627420070742) q[5];
ry(0.00037548384380129053) q[6];
rz(1.42060173900878) q[6];
ry(-0.6152020170026005) q[7];
rz(-2.8190176553926762) q[7];
ry(3.1363667193883855) q[8];
rz(-1.8232185828804246) q[8];
ry(-1.568643431625497) q[9];
rz(1.1787695484219345) q[9];
ry(-1.2640756895107588) q[10];
rz(-0.6177889087412023) q[10];
ry(1.9052796508637702) q[11];
rz(-2.539664649687235) q[11];
ry(-0.8988456432868391) q[12];
rz(-1.756347301456319) q[12];
ry(-2.308185917931732) q[13];
rz(-2.071191963032935) q[13];
ry(2.0711237953497497) q[14];
rz(2.225168442546174) q[14];
ry(1.7383969927271423) q[15];
rz(1.143298125385375) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.5242705656282189) q[0];
rz(-0.03161404420809606) q[0];
ry(0.5092125770051394) q[1];
rz(0.4953142806482163) q[1];
ry(-1.7595409229518477) q[2];
rz(1.3889984966895368) q[2];
ry(-0.000988866738197849) q[3];
rz(-1.2285333564221244) q[3];
ry(0.14118294924676356) q[4];
rz(1.32727798022366) q[4];
ry(-3.0745309874222775) q[5];
rz(0.9218119769395337) q[5];
ry(-3.132792994880609) q[6];
rz(-2.5693324670341124) q[6];
ry(1.3047371080196326e-05) q[7];
rz(3.0967861795091145) q[7];
ry(1.5709880271207615) q[8];
rz(1.6194735906017286) q[8];
ry(0.007013509208849023) q[9];
rz(-3.1357534526703232) q[9];
ry(2.067774598790738) q[10];
rz(-2.307890062704369) q[10];
ry(-2.533362181493914) q[11];
rz(-2.3795262095141356) q[11];
ry(1.064233680198718) q[12];
rz(0.8840732066505443) q[12];
ry(0.3201926302373357) q[13];
rz(0.4567005046733554) q[13];
ry(0.10368822032402421) q[14];
rz(0.6293061594179756) q[14];
ry(2.1549166677908165) q[15];
rz(2.4316397913929113) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.8328823208291292) q[0];
rz(-1.1859897237741555) q[0];
ry(2.9116814909980038) q[1];
rz(-1.2860173361418141) q[1];
ry(-2.922996276478114) q[2];
rz(-0.8192466587248932) q[2];
ry(-3.1405497269693803) q[3];
rz(-0.34772693780150143) q[3];
ry(-0.023666026558291087) q[4];
rz(0.35664569289501313) q[4];
ry(3.123713034428397) q[5];
rz(1.0255114911966225) q[5];
ry(6.160727327131355e-05) q[6];
rz(2.2431802368729823) q[6];
ry(1.0926271718368366) q[7];
rz(-3.0992241027134333) q[7];
ry(3.0615142301877087) q[8];
rz(-1.5115257944715603) q[8];
ry(2.5663640893020436) q[9];
rz(-0.4437657104441026) q[9];
ry(3.1389413562478174) q[10];
rz(-2.587322716670337) q[10];
ry(0.4452699703381784) q[11];
rz(1.0928345597624236) q[11];
ry(0.6144836789500918) q[12];
rz(-2.193640292547242) q[12];
ry(-1.4044241651925014) q[13];
rz(2.71439337520351) q[13];
ry(0.3546792118744735) q[14];
rz(-1.152949320010313) q[14];
ry(2.5401341593990834) q[15];
rz(-0.4226935178876321) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.3236299032481194) q[0];
rz(0.6079500315954602) q[0];
ry(2.821779699415996) q[1];
rz(-1.5896313849925354) q[1];
ry(-0.0003618789404926659) q[2];
rz(1.2271337089769745) q[2];
ry(-0.0024733057707038266) q[3];
rz(-0.060489028495461436) q[3];
ry(0.8011057069517562) q[4];
rz(0.6456877492320432) q[4];
ry(3.1012208018182665) q[5];
rz(1.335945690660625) q[5];
ry(-0.008796784247152358) q[6];
rz(-0.9752191923049002) q[6];
ry(3.140839413676345) q[7];
rz(2.8464453757088766) q[7];
ry(1.5680234110338527) q[8];
rz(-0.03925674524868341) q[8];
ry(1.5926217399225084) q[9];
rz(-0.035012030415889184) q[9];
ry(2.0176790591894562) q[10];
rz(0.7660946954070535) q[10];
ry(-1.8099362294020434) q[11];
rz(2.683453683529619) q[11];
ry(2.7705596751539026) q[12];
rz(-0.5126149242739881) q[12];
ry(-1.022464677863478) q[13];
rz(1.0819356434883245) q[13];
ry(1.5340638344956492) q[14];
rz(2.572262149417075) q[14];
ry(1.9971451555940651) q[15];
rz(-1.076067613423012) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.21057389833824125) q[0];
rz(3.0018079381667175) q[0];
ry(-1.284268849221676) q[1];
rz(-1.52722396072123) q[1];
ry(1.8146965703757052) q[2];
rz(1.2883939944052545) q[2];
ry(3.1399463127972056) q[3];
rz(3.0418428461211215) q[3];
ry(-3.138359182556487) q[4];
rz(-2.7437796093310287) q[4];
ry(-0.0032899073438551057) q[5];
rz(-1.8313928948646678) q[5];
ry(1.8963180049154666) q[6];
rz(2.02368300493353) q[6];
ry(-1.5690731352099576) q[7];
rz(0.00478172614031891) q[7];
ry(3.1396325987255476) q[8];
rz(0.28377837742811834) q[8];
ry(0.004924674328001365) q[9];
rz(1.366941967016256) q[9];
ry(2.0668770607031686) q[10];
rz(-1.892344289437597) q[10];
ry(0.00011681867127855659) q[11];
rz(2.719413224898905) q[11];
ry(-1.139962779667906) q[12];
rz(2.0997751854381748) q[12];
ry(2.984725190913775) q[13];
rz(-2.8853326976998432) q[13];
ry(-2.6360688956715412) q[14];
rz(1.1755961296750037) q[14];
ry(1.425333382998069) q[15];
rz(1.255122673929582) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(3.0495535012141746) q[0];
rz(0.3921034052334944) q[0];
ry(0.052133478847677495) q[1];
rz(-0.295739185246897) q[1];
ry(-0.0045462138914561905) q[2];
rz(-1.2020051419249635) q[2];
ry(-3.141464197891667) q[3];
rz(-2.495187079209746) q[3];
ry(0.06388449834048782) q[4];
rz(-2.6905513422662164) q[4];
ry(-3.13660027069705) q[5];
rz(-2.97673191564554) q[5];
ry(-3.1411490119951453) q[6];
rz(-1.300000050472514) q[6];
ry(0.00042787109210173924) q[7];
rz(3.1188861389420306) q[7];
ry(3.1413633182171923) q[8];
rz(2.258295570101063) q[8];
ry(-1.545487210266284) q[9];
rz(2.33309127260612) q[9];
ry(3.1412453970901177) q[10];
rz(1.0738566074794278) q[10];
ry(2.0579973002697516) q[11];
rz(-0.7446546544153447) q[11];
ry(0.5882292559670992) q[12];
rz(-1.64908994215366) q[12];
ry(-0.6556039466755932) q[13];
rz(0.7201389678486494) q[13];
ry(-0.7421228385719825) q[14];
rz(0.3404180288580241) q[14];
ry(1.2484496346493144) q[15];
rz(2.591634806279097) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(2.1339647386349148) q[0];
rz(-0.726960076050823) q[0];
ry(-1.6541581499997067) q[1];
rz(1.5631950045730272) q[1];
ry(1.3218406918008718) q[2];
rz(-0.39412197821334516) q[2];
ry(3.1412055680421545) q[3];
rz(-2.2376879735393294) q[3];
ry(-1.587389809014943) q[4];
rz(-3.1311036121758673) q[4];
ry(0.7425145213174167) q[5];
rz(-3.1364927832341345) q[5];
ry(-1.819756975937831) q[6];
rz(-2.8998624167930056) q[6];
ry(-0.09053142769417619) q[7];
rz(-2.0811392039432235) q[7];
ry(1.5613331238604469) q[8];
rz(-0.134207055625454) q[8];
ry(-0.003778410049774195) q[9];
rz(-2.8535845425721984) q[9];
ry(1.873780856069324) q[10];
rz(3.134062206723674) q[10];
ry(-3.139074725642079) q[11];
rz(0.005186595153545248) q[11];
ry(-1.7597802310909794) q[12];
rz(0.14588295487057668) q[12];
ry(-1.1743029966691572) q[13];
rz(1.3680101967427885) q[13];
ry(-2.577417732144322) q[14];
rz(0.4411396682979733) q[14];
ry(2.1460723112609803) q[15];
rz(-2.8483575526007714) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.699307527997294) q[0];
rz(-0.8113920899495186) q[0];
ry(0.47649554035660113) q[1];
rz(-0.35373109877188286) q[1];
ry(0.0009741879277109788) q[2];
rz(2.25582483300521) q[2];
ry(-3.1406004276868575) q[3];
rz(0.5643257217043017) q[3];
ry(1.4320883322063669) q[4];
rz(-0.06276511285559003) q[4];
ry(1.5714028771860855) q[5];
rz(-2.3182554454196516) q[5];
ry(-3.13962604752406) q[6];
rz(0.8051603243384173) q[6];
ry(-0.0006475081707510267) q[7];
rz(1.57230208626848) q[7];
ry(-0.008383320431029876) q[8];
rz(0.1363182469354481) q[8];
ry(1.6456805998626076) q[9];
rz(-3.090789148381346) q[9];
ry(0.0022071369650156214) q[10];
rz(0.7122450349793416) q[10];
ry(-2.4240483179119727) q[11];
rz(2.4709498928480493) q[11];
ry(1.5258075638349875) q[12];
rz(1.4542533822161674) q[12];
ry(-0.39878483549925575) q[13];
rz(0.15289137694359756) q[13];
ry(-1.6251497620393272) q[14];
rz(-2.7208729113736903) q[14];
ry(1.9664739323102691) q[15];
rz(-1.8176713786344523) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(2.4992085619501774) q[0];
rz(-1.7487811416052748) q[0];
ry(-2.5796034835074786) q[1];
rz(-2.0620208730344176) q[1];
ry(-3.0671038872653074) q[2];
rz(-1.4872823148907126) q[2];
ry(-1.5694327110678632) q[3];
rz(-2.608659239946173) q[3];
ry(2.1041109627658474) q[4];
rz(-1.6054438845973773) q[4];
ry(-1.2218519730316593) q[5];
rz(-2.1619824152299936) q[5];
ry(-0.08293951696766606) q[6];
rz(2.337543863273158) q[6];
ry(-2.258937015846589) q[7];
rz(2.2656986592124584) q[7];
ry(1.6746406174985164) q[8];
rz(-0.811986315150124) q[8];
ry(3.1331838234392433) q[9];
rz(1.611879176160758) q[9];
ry(1.8468048594310718) q[10];
rz(-0.5031918486823953) q[10];
ry(-3.1412659351141907) q[11];
rz(-0.8490124926977964) q[11];
ry(-0.2210861609571806) q[12];
rz(1.626164733902745) q[12];
ry(0.6646086336189417) q[13];
rz(-0.49335367953319764) q[13];
ry(-1.8758018337222229) q[14];
rz(-1.038719981954289) q[14];
ry(0.532211512266367) q[15];
rz(-0.8556838804035078) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-2.681135420353373) q[0];
rz(0.009690760323344363) q[0];
ry(1.6348133579936996) q[1];
rz(1.4251288650323737) q[1];
ry(0.00028027055251733657) q[2];
rz(-2.2647340473329916) q[2];
ry(-3.1389220353497898) q[3];
rz(2.433254596971734) q[3];
ry(-1.567698511837156) q[4];
rz(2.5169029579569724) q[4];
ry(-3.141535532442502) q[5];
rz(2.9040583472482133) q[5];
ry(-0.0022446323647161764) q[6];
rz(0.5537279810604583) q[6];
ry(-3.140782056283157) q[7];
rz(0.6845088631866919) q[7];
ry(0.005398129200377678) q[8];
rz(-0.9927845795104423) q[8];
ry(0.3489537337259008) q[9];
rz(-1.2300137715482748) q[9];
ry(-2.9534455078616673) q[10];
rz(1.355399444241004) q[10];
ry(0.8898285694519269) q[11];
rz(0.39490111889421353) q[11];
ry(1.9755847334010779) q[12];
rz(2.649078437942561) q[12];
ry(2.2872404169974154) q[13];
rz(-3.06079466473842) q[13];
ry(-1.7020422544403726) q[14];
rz(-0.8762168932810893) q[14];
ry(0.5600377700109673) q[15];
rz(1.7081727697576286) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.5127046933382742) q[0];
rz(-3.111489474332621) q[0];
ry(-2.0328229839533876) q[1];
rz(1.2383287361323978) q[1];
ry(-3.1412446835098904) q[2];
rz(-2.678727295978897) q[2];
ry(-0.2970188711153128) q[3];
rz(2.9431070933949015) q[3];
ry(1.523058325559118) q[4];
rz(-1.5384940175514596) q[4];
ry(1.5695436007101424) q[5];
rz(1.4961469087644914) q[5];
ry(-6.6065619717115e-05) q[6];
rz(1.7416101961084733) q[6];
ry(1.5571717771229148) q[7];
rz(0.7174418895565822) q[7];
ry(1.5668296894682108) q[8];
rz(-3.0656446277009817) q[8];
ry(0.008821817876907724) q[9];
rz(-1.0395119221719036) q[9];
ry(0.9708235999147418) q[10];
rz(2.642559061020887) q[10];
ry(0.0027639682382640768) q[11];
rz(1.5447048984600413) q[11];
ry(-0.8217038400335941) q[12];
rz(0.6981982711508286) q[12];
ry(2.4075675903758222) q[13];
rz(0.6621050829654417) q[13];
ry(1.847555612394591) q[14];
rz(-2.4245363653807814) q[14];
ry(1.805651127191486) q[15];
rz(-0.464145762313609) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-2.2924688725350233) q[0];
rz(2.235456701621035) q[0];
ry(3.0905229036241457) q[1];
rz(-1.483048194677187) q[1];
ry(3.139849743332599) q[2];
rz(2.9520667361038977) q[2];
ry(-1.5121829577857682e-05) q[3];
rz(0.5389931778675159) q[3];
ry(1.5745344867674795) q[4];
rz(2.23110150597767) q[4];
ry(-3.1412908822748364) q[5];
rz(2.0978799595252933) q[5];
ry(2.1756014054058515) q[6];
rz(3.140455634437889) q[6];
ry(0.1789844191768699) q[7];
rz(-3.1392894368225956) q[7];
ry(-3.1326514422161362) q[8];
rz(-3.1127690437942532) q[8];
ry(0.0003772102554275847) q[9];
rz(-2.233097099819732) q[9];
ry(-3.141501775821803) q[10];
rz(0.634835913474242) q[10];
ry(2.905574417490796) q[11];
rz(3.1002789983774637) q[11];
ry(-1.5009717125346649) q[12];
rz(1.4017283853806342) q[12];
ry(-2.880680139869669) q[13];
rz(-0.5863297578567341) q[13];
ry(-2.720463667291761) q[14];
rz(-2.0067575420467056) q[14];
ry(-1.6680927701388724) q[15];
rz(3.0159505435530503) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(2.707181113789832) q[0];
rz(-1.6880371746942597) q[0];
ry(-0.6795723653632475) q[1];
rz(-0.07245827835779665) q[1];
ry(-2.978648495434704) q[2];
rz(0.05777836285563499) q[2];
ry(-2.9539837226297188) q[3];
rz(0.0407104045548761) q[3];
ry(-0.6461211249870011) q[4];
rz(0.541897075879441) q[4];
ry(-0.3070170962722517) q[5];
rz(0.6646662756184758) q[5];
ry(-2.5159665456257154) q[6];
rz(0.0015202829286673494) q[6];
ry(-2.835817490234606) q[7];
rz(-1.6980442273795284) q[7];
ry(0.0009677824380638356) q[8];
rz(-0.2861859196023898) q[8];
ry(-1.444198597264748) q[9];
rz(-2.2349637884333697) q[9];
ry(-1.191616802088593) q[10];
rz(-0.6875288739219805) q[10];
ry(-3.1405604667909355) q[11];
rz(-2.2456536892363212) q[11];
ry(-3.027170567045008) q[12];
rz(0.5381885958504) q[12];
ry(-1.3699749471073956) q[13];
rz(2.724315173915846) q[13];
ry(1.2733708483022697) q[14];
rz(-2.9373647077681553) q[14];
ry(-1.8933917416773023) q[15];
rz(2.9359811986851656) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.7928042357129925) q[0];
rz(2.3693552591261393) q[0];
ry(1.578474996608729) q[1];
rz(-3.1133297464071474) q[1];
ry(1.5671537993949114) q[2];
rz(1.855414283867066) q[2];
ry(3.1414725244040795) q[3];
rz(-0.6187148428081652) q[3];
ry(-3.1403565923140344) q[4];
rz(2.1152489227136058) q[4];
ry(-3.141308948697737) q[5];
rz(-1.860038811854742) q[5];
ry(0.8766987104438323) q[6];
rz(3.1380664181911344) q[6];
ry(-3.1408125286041724) q[7];
rz(-0.12674428074975627) q[7];
ry(-3.140159792563104) q[8];
rz(-1.904643907041371) q[8];
ry(-3.1288277873546235) q[9];
rz(-1.204251552106023) q[9];
ry(-3.140314976594624) q[10];
rz(-3.0312192467433) q[10];
ry(1.011382912509504e-05) q[11];
rz(-1.2327660623504537) q[11];
ry(-1.5949259468779342) q[12];
rz(0.06729757230042353) q[12];
ry(1.7579628091786796) q[13];
rz(-0.06497397903216129) q[13];
ry(-2.432639692321128) q[14];
rz(-1.7123686805817284) q[14];
ry(-1.8467233595331782) q[15];
rz(0.03671029877963151) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(3.527804350468783e-05) q[0];
rz(-1.0459987966530848) q[0];
ry(-2.33414168428735) q[1];
rz(1.9150045273959848) q[1];
ry(-0.051169879810501584) q[2];
rz(-0.18691181455876296) q[2];
ry(-1.572350587199871) q[3];
rz(-0.012872832460346649) q[3];
ry(1.5712778959708327) q[4];
rz(2.1063235504985673) q[4];
ry(1.8224978324841627) q[5];
rz(1.2698238261254031) q[5];
ry(0.9804965206482441) q[6];
rz(-1.4618713097688474) q[6];
ry(-1.5699209252464803) q[7];
rz(0.15142581923101428) q[7];
ry(1.5693077320360957) q[8];
rz(0.39565423572192604) q[8];
ry(-0.23363841855132034) q[9];
rz(-2.59002682511448) q[9];
ry(-0.2891373510994999) q[10];
rz(1.8151668928542957) q[10];
ry(1.5722695721237603) q[11];
rz(0.0049221969955697276) q[11];
ry(-1.9465174114580828) q[12];
rz(1.319887157773442) q[12];
ry(-3.083370798934761) q[13];
rz(0.3092511162432538) q[13];
ry(0.9640796149143913) q[14];
rz(-1.3357738774689611) q[14];
ry(-2.404307349217555) q[15];
rz(3.07781795127602) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.024478366713697106) q[0];
rz(0.0755496161638143) q[0];
ry(0.0008854162949054613) q[1];
rz(2.719158549408748) q[1];
ry(-1.5892077141131997) q[2];
rz(1.8325359542389803) q[2];
ry(-0.003949730395275353) q[3];
rz(-0.03968690764065814) q[3];
ry(0.002719898971944801) q[4];
rz(-1.455010814749371) q[4];
ry(0.0005437205155356395) q[5];
rz(-1.952993656035002) q[5];
ry(0.0006627381338167581) q[6];
rz(1.311228830171749) q[6];
ry(2.9483677408842555) q[7];
rz(2.9468562166679173) q[7];
ry(-0.000505341300103099) q[8];
rz(-1.9667925409980258) q[8];
ry(1.6956737929876873) q[9];
rz(-1.4336693004662282) q[9];
ry(0.00021662568581337638) q[10];
rz(2.163023700710378) q[10];
ry(3.133824845887371) q[11];
rz(-3.1373250459989075) q[11];
ry(1.7074232837701793) q[12];
rz(0.47099922584477744) q[12];
ry(0.000825213184083573) q[13];
rz(0.9919338433896342) q[13];
ry(-2.174586694538229) q[14];
rz(3.0892788138929377) q[14];
ry(1.0557968776092421) q[15];
rz(-1.443553010403111) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.7400790192358064) q[0];
rz(2.7894717325389777) q[0];
ry(3.090253084097082) q[1];
rz(3.0108980788823763) q[1];
ry(-0.19268891737945193) q[2];
rz(-1.5022146111733772) q[2];
ry(-3.1250982650789396) q[3];
rz(3.0893128005575714) q[3];
ry(-0.010518545734980678) q[4];
rz(-0.6466980358081029) q[4];
ry(-0.0021767021858420423) q[5];
rz(0.44101909354877683) q[5];
ry(2.2401757311385495e-06) q[6];
rz(1.8579441920385031) q[6];
ry(0.0005714170123400919) q[7];
rz(0.1931810918712893) q[7];
ry(1.5697358207341208) q[8];
rz(-2.216161416783554) q[8];
ry(3.141246983899318) q[9];
rz(-0.3365944060425239) q[9];
ry(0.48885987135011605) q[10];
rz(-2.011459708932799) q[10];
ry(-1.571196910996593) q[11];
rz(1.3526833972745562) q[11];
ry(-1.0772938320119607) q[12];
rz(1.2872251993927892) q[12];
ry(0.1732933330191552) q[13];
rz(0.872902206994091) q[13];
ry(1.8693879911174622) q[14];
rz(1.0060490470755103) q[14];
ry(-1.7050297202184534) q[15];
rz(-0.3000377733972991) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.1943166804928502) q[0];
rz(-2.7810467796341927) q[0];
ry(-3.128919391736622) q[1];
rz(-2.6220092714905046) q[1];
ry(2.1028825776027844) q[2];
rz(1.4291296630706258) q[2];
ry(-2.296479460640624) q[3];
rz(-2.9804447673877665) q[3];
ry(1.5726702810212125) q[4];
rz(2.9930861458185873) q[4];
ry(3.140434990074393) q[5];
rz(-0.4252185806500952) q[5];
ry(-3.1414991674313937) q[6];
rz(1.7249740600653456) q[6];
ry(-2.9482451617463084) q[7];
rz(0.04167896361032817) q[7];
ry(3.1414806036391663) q[8];
rz(0.6595848027465613) q[8];
ry(1.5723595822655234) q[9];
rz(1.5730590355626497) q[9];
ry(0.011294355945777212) q[10];
rz(1.677006048912017) q[10];
ry(-0.0020275284176011366) q[11];
rz(0.17820544072237698) q[11];
ry(-1.751649564930234) q[12];
rz(-2.108287590869616) q[12];
ry(-2.9457111993686356) q[13];
rz(1.2861742361422812) q[13];
ry(-2.6579942608036897) q[14];
rz(-2.103093285354574) q[14];
ry(-0.39565372546182864) q[15];
rz(0.82808902402454) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.02066048880275151) q[0];
rz(0.19583515838060886) q[0];
ry(-3.1404636827843633) q[1];
rz(-2.3506708409564268) q[1];
ry(3.141067864307082) q[2];
rz(-1.2379340959159693) q[2];
ry(-3.1320604527719063) q[3];
rz(0.16125763586574673) q[3];
ry(-3.1380103802783363) q[4];
rz(-2.782666829741134) q[4];
ry(-1.5723208266379336) q[5];
rz(-0.7814024856843623) q[5];
ry(1.5641182998110068) q[6];
rz(-2.9348023789127655) q[6];
ry(-1.586338959919897) q[7];
rz(-2.4668604265814738) q[7];
ry(3.0256975428536474) q[8];
rz(-3.0539691475639947) q[8];
ry(-1.5719493152161572) q[9];
rz(0.26576907498959584) q[9];
ry(-2.074619742083225) q[10];
rz(1.4870632405511346) q[10];
ry(1.7026704204596448) q[11];
rz(2.6764422579904528) q[11];
ry(0.709326093821278) q[12];
rz(2.787764825366298) q[12];
ry(0.6141832382651607) q[13];
rz(1.0950296052797672) q[13];
ry(1.1914650381923249) q[14];
rz(-1.931040668607272) q[14];
ry(1.5535069668282635) q[15];
rz(-1.5409285680595108) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.29770572063719125) q[0];
rz(-0.3385978665410748) q[0];
ry(-3.127825213671513) q[1];
rz(-3.053299507389541) q[1];
ry(-1.2452569060092198) q[2];
rz(-0.5264971275060466) q[2];
ry(-1.568765390531941) q[3];
rz(-1.5689545982898077) q[3];
ry(0.0014707963776281474) q[4];
rz(-0.19630160196069554) q[4];
ry(0.000517008962857106) q[5];
rz(0.8111605703273357) q[5];
ry(-3.1169348250825237) q[6];
rz(1.5126606453190625) q[6];
ry(-0.8160564314973069) q[7];
rz(2.6605906229580913) q[7];
ry(0.11317834361821427) q[8];
rz(0.3645194306573201) q[8];
ry(3.0957885695796388) q[9];
rz(1.4258777437821015) q[9];
ry(0.002603910560451237) q[10];
rz(0.7357336602003295) q[10];
ry(-3.1404018881232636) q[11];
rz(0.8168809154964235) q[11];
ry(-1.946487776105198) q[12];
rz(1.6339824293494853) q[12];
ry(2.6766373344293606) q[13];
rz(3.1132033411780493) q[13];
ry(-1.880044546095449) q[14];
rz(1.9095447126412861) q[14];
ry(2.7276822336903357) q[15];
rz(-1.416742599898189) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(2.9391519941289515) q[0];
rz(-1.1225227097408519) q[0];
ry(1.571348819023168) q[1];
rz(-1.1992125638577198) q[1];
ry(-1.5704977577422587) q[2];
rz(1.5710210707836747) q[2];
ry(-1.5709033634321603) q[3];
rz(1.7374180887571633) q[3];
ry(0.005108596361640417) q[4];
rz(-1.1010573809572077) q[4];
ry(-0.018479246196874577) q[5];
rz(0.07446323588271472) q[5];
ry(3.134453951801451) q[6];
rz(2.542626872800034) q[6];
ry(-3.1360428150505553) q[7];
rz(2.4843256402941822) q[7];
ry(-3.0775697205861547) q[8];
rz(0.21971553049359865) q[8];
ry(-3.1415831473348033) q[9];
rz(-0.5193523764897773) q[9];
ry(1.717473010304837) q[10];
rz(-1.199266908742727) q[10];
ry(-0.9114128109809986) q[11];
rz(-1.7662495046517623) q[11];
ry(1.6140378948881677) q[12];
rz(1.991532432225122) q[12];
ry(-2.84183683471618) q[13];
rz(-1.469147030575245) q[13];
ry(1.0853652383277803) q[14];
rz(-2.0196695952412798) q[14];
ry(1.5749915849999163) q[15];
rz(2.617192387297386) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.5328606493740018) q[0];
rz(0.4634796336355223) q[0];
ry(3.1377154921491894) q[1];
rz(-1.9340023738365506) q[1];
ry(-1.5705419827573035) q[2];
rz(2.1162182309877804) q[2];
ry(3.1415758196100096) q[3];
rz(1.3455201212838546) q[3];
ry(3.1414047556920006) q[4];
rz(-2.3572093645667165) q[4];
ry(-3.141509090668388) q[5];
rz(-3.048444232799939) q[5];
ry(-0.00552628797159881) q[6];
rz(-2.6701636794340042) q[6];
ry(0.9527957069337238) q[7];
rz(-1.4938563874687931) q[7];
ry(-0.43636920571117427) q[8];
rz(0.5352033434389076) q[8];
ry(3.117897776343839) q[9];
rz(-2.7153349797326105) q[9];
ry(0.0006212290118847719) q[10];
rz(3.1269806048726254) q[10];
ry(0.013465494586567183) q[11];
rz(2.720054534921359) q[11];
ry(0.006184223282119206) q[12];
rz(1.1497544001141806) q[12];
ry(-0.004867309572867739) q[13];
rz(-1.7315077748073637) q[13];
ry(-1.247430935232872) q[14];
rz(0.4183645185782542) q[14];
ry(0.10268208481759168) q[15];
rz(0.49217258773891415) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-3.1412807085119168) q[0];
rz(2.05788979974444) q[0];
ry(3.140663646557293) q[1];
rz(3.0110439327757743) q[1];
ry(-0.00017224855488254378) q[2];
rz(-1.961871762349747) q[2];
ry(-3.1377083265023917) q[3];
rz(2.1745145518324325) q[3];
ry(1.5824922580524285) q[4];
rz(-3.1278795478589623) q[4];
ry(-2.4420688056285087) q[5];
rz(-1.5867788380903338) q[5];
ry(-0.0032023491530904524) q[6];
rz(3.0015527285351293) q[6];
ry(0.01275091074460286) q[7];
rz(0.3374641754260148) q[7];
ry(1.725036335570401) q[8];
rz(-3.0485879599216426) q[8];
ry(3.1415730183349906) q[9];
rz(2.7928649529232885) q[9];
ry(2.917272969350065) q[10];
rz(-1.7979913884065395) q[10];
ry(0.4141315498511789) q[11];
rz(-2.2617915989110875) q[11];
ry(1.5898173762752448) q[12];
rz(2.1387935778886744) q[12];
ry(-0.3947856468386961) q[13];
rz(-1.1597639766506804) q[13];
ry(1.8191804815594874) q[14];
rz(-2.302259666730804) q[14];
ry(1.5976296921947006) q[15];
rz(-0.4162259546101508) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.0107939737053968) q[0];
rz(-1.4641317403400773) q[0];
ry(3.139156962359244) q[1];
rz(2.175026378865014) q[1];
ry(-0.0021926046672925957) q[2];
rz(1.4053824358169271) q[2];
ry(2.053359015227686e-05) q[3];
rz(2.1294314592731878) q[3];
ry(-0.0003057495814502431) q[4];
rz(0.7144836885733618) q[4];
ry(3.036758368919586) q[5];
rz(-0.007626551337734711) q[5];
ry(-2.9133989765778856) q[6];
rz(-1.0022081205309155) q[6];
ry(2.981122045078722) q[7];
rz(0.6559190040323291) q[7];
ry(-0.40688979306992307) q[8];
rz(3.122366840099527) q[8];
ry(-3.1405789788611185) q[9];
rz(0.7289383604495638) q[9];
ry(-3.1404243431381587) q[10];
rz(-2.8354354196293894) q[10];
ry(3.129415429558612) q[11];
rz(-2.8863160815739315) q[11];
ry(1.5018813310764418) q[12];
rz(0.01849905904840021) q[12];
ry(-2.6858670448760917) q[13];
rz(-0.10690766926251397) q[13];
ry(1.344694422087995) q[14];
rz(0.9018461278147836) q[14];
ry(2.689350147293366) q[15];
rz(1.5358707276482333) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-3.1410624098222) q[0];
rz(1.7095231610304022) q[0];
ry(-1.5707121938211541) q[1];
rz(-1.7008443009401244) q[1];
ry(-1.5710707933522439) q[2];
rz(-0.012452584990076487) q[2];
ry(1.5723613073925842) q[3];
rz(-0.12731519320545356) q[3];
ry(0.002490247493386203) q[4];
rz(2.4026409971970093) q[4];
ry(-1.5751097621336418) q[5];
rz(-0.10618821319766879) q[5];
ry(0.005883448963254878) q[6];
rz(2.53815431029755) q[6];
ry(-3.13992948739022) q[7];
rz(-0.6793831328416461) q[7];
ry(1.5704557599631122) q[8];
rz(1.5541218562587593) q[8];
ry(0.0042828395337108915) q[9];
rz(2.966164525117951) q[9];
ry(1.5067537516612115) q[10];
rz(-0.15275694292547645) q[10];
ry(-1.5358124713692005) q[11];
rz(1.0583874978567325) q[11];
ry(3.1406382006345237) q[12];
rz(-1.342696943094553) q[12];
ry(-0.0051126199882087775) q[13];
rz(1.6022038936964502) q[13];
ry(-1.5673776598456577) q[14];
rz(-2.9359968738800934) q[14];
ry(-3.1381246790038504) q[15];
rz(-2.3912329760864224) q[15];