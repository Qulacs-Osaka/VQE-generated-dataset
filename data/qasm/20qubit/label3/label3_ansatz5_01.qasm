OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.5978742059858577) q[0];
ry(1.6423240669527606) q[1];
cx q[0],q[1];
ry(-2.121081070336449) q[0];
ry(1.5944768088249939) q[1];
cx q[0],q[1];
ry(2.776530057815593) q[2];
ry(-2.142397854815118) q[3];
cx q[2],q[3];
ry(3.053271515519281) q[2];
ry(-0.1245232235258315) q[3];
cx q[2],q[3];
ry(1.8586627779188305) q[4];
ry(1.7263339911048643) q[5];
cx q[4],q[5];
ry(-0.21465887681929008) q[4];
ry(-2.1601455508141987) q[5];
cx q[4],q[5];
ry(-1.3337575032014268) q[6];
ry(1.0846998359583022) q[7];
cx q[6],q[7];
ry(0.4568949631507316) q[6];
ry(-3.053611559082931) q[7];
cx q[6],q[7];
ry(-0.2399926804360426) q[8];
ry(-2.5106108208697226) q[9];
cx q[8],q[9];
ry(1.7705141639371975) q[8];
ry(1.5844394904279149) q[9];
cx q[8],q[9];
ry(1.8452548287194155) q[10];
ry(-0.2905099766669927) q[11];
cx q[10],q[11];
ry(2.665092823917041) q[10];
ry(-1.062716707348625) q[11];
cx q[10],q[11];
ry(1.1675921991345497) q[12];
ry(-1.8438908493262005) q[13];
cx q[12],q[13];
ry(2.8313008073717953) q[12];
ry(-1.1730702441282346) q[13];
cx q[12],q[13];
ry(1.7647760356029671) q[14];
ry(2.038657187422938) q[15];
cx q[14],q[15];
ry(-1.094680511863717) q[14];
ry(-2.0600748237251816) q[15];
cx q[14],q[15];
ry(3.116419292329061) q[16];
ry(-2.9092998533687897) q[17];
cx q[16],q[17];
ry(-0.7827892697157521) q[16];
ry(2.8326586369166558) q[17];
cx q[16],q[17];
ry(-1.1685325945236953) q[18];
ry(-1.6173802630841427) q[19];
cx q[18],q[19];
ry(1.015928473469545) q[18];
ry(2.1521863067620224) q[19];
cx q[18],q[19];
ry(0.5986946000716266) q[1];
ry(-3.068421306304389) q[2];
cx q[1],q[2];
ry(2.0850692549119465) q[1];
ry(-2.141365091426337) q[2];
cx q[1],q[2];
ry(-0.4319007880120626) q[3];
ry(-0.5693226776402882) q[4];
cx q[3],q[4];
ry(-1.198058996855656) q[3];
ry(1.5613096606145425) q[4];
cx q[3],q[4];
ry(-3.1125497567244973) q[5];
ry(-0.884019242623403) q[6];
cx q[5],q[6];
ry(1.1813172164778736) q[5];
ry(-2.162045166764379) q[6];
cx q[5],q[6];
ry(1.6833358663040743) q[7];
ry(-2.591857553666454) q[8];
cx q[7],q[8];
ry(-1.4015752039495157) q[7];
ry(-0.3395835906991982) q[8];
cx q[7],q[8];
ry(-1.3749023419192898) q[9];
ry(2.3390408861263574) q[10];
cx q[9],q[10];
ry(-3.129438821122522) q[9];
ry(0.05232910508782129) q[10];
cx q[9],q[10];
ry(2.2970855127080987) q[11];
ry(-0.6028784884788614) q[12];
cx q[11],q[12];
ry(-0.2602031855620246) q[11];
ry(-0.6477209553968647) q[12];
cx q[11],q[12];
ry(-2.0061848631221304) q[13];
ry(-0.5290854710955184) q[14];
cx q[13],q[14];
ry(1.4494854790786196) q[13];
ry(-2.578027833592301) q[14];
cx q[13],q[14];
ry(-0.4969920879272696) q[15];
ry(2.6480567512787108) q[16];
cx q[15],q[16];
ry(-2.121555685143151) q[15];
ry(2.5724084735463797) q[16];
cx q[15],q[16];
ry(3.0217830310510667) q[17];
ry(0.06412349364585965) q[18];
cx q[17],q[18];
ry(-1.3542729444207087) q[17];
ry(2.632469167492098) q[18];
cx q[17],q[18];
ry(1.0170958772613252) q[0];
ry(-0.2106455014685098) q[1];
cx q[0],q[1];
ry(2.535776823976203) q[0];
ry(-2.539934156849919) q[1];
cx q[0],q[1];
ry(-2.130476483468949) q[2];
ry(1.2580127351003814) q[3];
cx q[2],q[3];
ry(3.034434132050541) q[2];
ry(1.9639162417206828) q[3];
cx q[2],q[3];
ry(-1.6061491427547376) q[4];
ry(-1.649714648457314) q[5];
cx q[4],q[5];
ry(2.3527463642894193) q[4];
ry(2.4381383380406594) q[5];
cx q[4],q[5];
ry(0.03380288407902356) q[6];
ry(0.0006273614800260674) q[7];
cx q[6],q[7];
ry(-1.5760452859550975) q[6];
ry(-1.5963571101567018) q[7];
cx q[6],q[7];
ry(-1.9980327166004652) q[8];
ry(-1.1126861434152018) q[9];
cx q[8],q[9];
ry(1.6389314933696237) q[8];
ry(2.9316544668425446) q[9];
cx q[8],q[9];
ry(1.681774908916541) q[10];
ry(-2.792297916436178) q[11];
cx q[10],q[11];
ry(1.450591230203083) q[10];
ry(2.3985555688180016) q[11];
cx q[10],q[11];
ry(0.02417384501097075) q[12];
ry(0.45302878786039164) q[13];
cx q[12],q[13];
ry(-2.546037933534848) q[12];
ry(1.523222149158677) q[13];
cx q[12],q[13];
ry(-0.0850452471371479) q[14];
ry(2.750261121990813) q[15];
cx q[14],q[15];
ry(-1.5815850723157465) q[14];
ry(-0.09152157950092932) q[15];
cx q[14],q[15];
ry(-2.2359411180957203) q[16];
ry(2.6610051889097215) q[17];
cx q[16],q[17];
ry(0.19302757285571026) q[16];
ry(1.590431363093829) q[17];
cx q[16],q[17];
ry(-2.313857439081968) q[18];
ry(0.26847233406053744) q[19];
cx q[18],q[19];
ry(-3.1079022284478848) q[18];
ry(-0.8514524872474579) q[19];
cx q[18],q[19];
ry(-2.6393433434903004) q[1];
ry(-2.526699184143687) q[2];
cx q[1],q[2];
ry(0.7230933379126452) q[1];
ry(2.6856162754318076) q[2];
cx q[1],q[2];
ry(3.0488498858243145) q[3];
ry(3.084309835949032) q[4];
cx q[3],q[4];
ry(1.5465721605441791) q[3];
ry(-1.5854185307553395) q[4];
cx q[3],q[4];
ry(3.103364765252383) q[5];
ry(0.7762877765682836) q[6];
cx q[5],q[6];
ry(0.37925257444055344) q[5];
ry(2.423022891462699) q[6];
cx q[5],q[6];
ry(1.382503852191875) q[7];
ry(-0.48248614919725263) q[8];
cx q[7],q[8];
ry(1.5928595345116783) q[7];
ry(-0.3189181677115802) q[8];
cx q[7],q[8];
ry(0.07511106646252777) q[9];
ry(0.11534112977914435) q[10];
cx q[9],q[10];
ry(0.4348187287625716) q[9];
ry(1.3584292299071041) q[10];
cx q[9],q[10];
ry(2.870314344688327) q[11];
ry(1.841932593121414) q[12];
cx q[11],q[12];
ry(0.3100818712717134) q[11];
ry(3.052761609004468) q[12];
cx q[11],q[12];
ry(3.040856401228982) q[13];
ry(1.3033883100876498) q[14];
cx q[13],q[14];
ry(-1.582835915538709) q[13];
ry(-1.7274539216929776) q[14];
cx q[13],q[14];
ry(-0.028473211508213962) q[15];
ry(0.09250064668516167) q[16];
cx q[15],q[16];
ry(-1.5908778889864035) q[15];
ry(-1.5552930868875992) q[16];
cx q[15],q[16];
ry(-2.375031756110824) q[17];
ry(-2.399741612951849) q[18];
cx q[17],q[18];
ry(-1.6488327180965123) q[17];
ry(0.9749549921808902) q[18];
cx q[17],q[18];
ry(-2.6630087725958447) q[0];
ry(-1.6225762694415904) q[1];
cx q[0],q[1];
ry(-0.583662919487976) q[0];
ry(0.11010982046711693) q[1];
cx q[0],q[1];
ry(-0.7144674231260469) q[2];
ry(0.22594213635481175) q[3];
cx q[2],q[3];
ry(-2.785788340877717) q[2];
ry(-1.6831631169272034) q[3];
cx q[2],q[3];
ry(1.2141498469249035) q[4];
ry(2.786439549136876) q[5];
cx q[4],q[5];
ry(-3.139824285186871) q[4];
ry(-2.6879123581575377) q[5];
cx q[4],q[5];
ry(3.1406352100289983) q[6];
ry(-2.9916413781385143) q[7];
cx q[6],q[7];
ry(-1.5685836223719836) q[6];
ry(1.6067034987958637) q[7];
cx q[6],q[7];
ry(1.652317566083915) q[8];
ry(3.1078544069574305) q[9];
cx q[8],q[9];
ry(-1.5040870844639045) q[8];
ry(1.6218150758926217) q[9];
cx q[8],q[9];
ry(-1.637561423753537) q[10];
ry(0.43644075942431476) q[11];
cx q[10],q[11];
ry(1.8186969306742284) q[10];
ry(0.4031253650751938) q[11];
cx q[10],q[11];
ry(1.687004004221605) q[12];
ry(-0.10849067362872829) q[13];
cx q[12],q[13];
ry(-2.5806177791282803) q[12];
ry(-2.4075317131457394) q[13];
cx q[12],q[13];
ry(0.2329542368161156) q[14];
ry(-2.44734780457544) q[15];
cx q[14],q[15];
ry(-0.16599072381489055) q[14];
ry(1.5378035646778967) q[15];
cx q[14],q[15];
ry(0.01588657214689086) q[16];
ry(1.9921732742411802) q[17];
cx q[16],q[17];
ry(-1.5465479853740298) q[16];
ry(0.508287928747317) q[17];
cx q[16],q[17];
ry(-1.8252363589379677) q[18];
ry(-0.5366061927244825) q[19];
cx q[18],q[19];
ry(1.7349550486176153) q[18];
ry(1.3477051383009346) q[19];
cx q[18],q[19];
ry(-2.2855244176610516) q[1];
ry(-3.1137248429220583) q[2];
cx q[1],q[2];
ry(1.5104651102815199) q[1];
ry(2.9434644373920222) q[2];
cx q[1],q[2];
ry(3.116739351952013) q[3];
ry(1.8616031710892997) q[4];
cx q[3],q[4];
ry(0.14819843254511955) q[3];
ry(-1.5859749210147203) q[4];
cx q[3],q[4];
ry(-0.32655717757884195) q[5];
ry(-0.06184851398981643) q[6];
cx q[5],q[6];
ry(-1.5199434281022208) q[5];
ry(-1.553798387636777) q[6];
cx q[5],q[6];
ry(2.739921485254125) q[7];
ry(0.9821561316612009) q[8];
cx q[7],q[8];
ry(-0.02168054971308647) q[7];
ry(-0.00556542948307559) q[8];
cx q[7],q[8];
ry(3.0435294983400865) q[9];
ry(-0.07193179119303128) q[10];
cx q[9],q[10];
ry(-0.41543022499939747) q[9];
ry(-0.3692045815935714) q[10];
cx q[9],q[10];
ry(2.9754759541009324) q[11];
ry(0.731190331032444) q[12];
cx q[11],q[12];
ry(1.6091000001591347) q[11];
ry(-1.711552330140594) q[12];
cx q[11],q[12];
ry(-2.109987937457161) q[13];
ry(1.587742777539627) q[14];
cx q[13],q[14];
ry(-2.197822159578216) q[13];
ry(0.06498330274103253) q[14];
cx q[13],q[14];
ry(-0.41556323750744095) q[15];
ry(-0.0252895002404953) q[16];
cx q[15],q[16];
ry(-2.741580635004942) q[15];
ry(2.6711082817473035) q[16];
cx q[15],q[16];
ry(-1.5896718640966698) q[17];
ry(-2.9551715275383996) q[18];
cx q[17],q[18];
ry(-1.5715855488778434) q[17];
ry(-0.9514739332518389) q[18];
cx q[17],q[18];
ry(-0.2629387404386856) q[0];
ry(2.5520771541078053) q[1];
cx q[0],q[1];
ry(1.2385019450904675) q[0];
ry(1.6954908308573584) q[1];
cx q[0],q[1];
ry(2.835015014786015) q[2];
ry(2.942522120256522) q[3];
cx q[2],q[3];
ry(-3.121305325973931) q[2];
ry(-0.5294640992805782) q[3];
cx q[2],q[3];
ry(-0.10395908354127076) q[4];
ry(-2.5371455197488384) q[5];
cx q[4],q[5];
ry(3.1212766517658865) q[4];
ry(3.0858420487425646) q[5];
cx q[4],q[5];
ry(3.127066392977206) q[6];
ry(-1.8187681846122286) q[7];
cx q[6],q[7];
ry(3.1389672241620623) q[6];
ry(-0.6688465468596562) q[7];
cx q[6],q[7];
ry(2.1862682437215213) q[8];
ry(-0.01687947337768738) q[9];
cx q[8],q[9];
ry(1.6445290850889882) q[8];
ry(1.644398789574156) q[9];
cx q[8],q[9];
ry(1.6934135801247616) q[10];
ry(-0.06850883616772724) q[11];
cx q[10],q[11];
ry(-0.30670098239278737) q[10];
ry(-0.16193649741724683) q[11];
cx q[10],q[11];
ry(2.278429058605502) q[12];
ry(-0.4631124748764126) q[13];
cx q[12],q[13];
ry(3.066730698973991) q[12];
ry(0.004519803189436011) q[13];
cx q[12],q[13];
ry(1.5972297648714484) q[14];
ry(1.2271034687839018) q[15];
cx q[14],q[15];
ry(-3.137887528193659) q[14];
ry(1.587062794490759) q[15];
cx q[14],q[15];
ry(3.1283088889390713) q[16];
ry(-1.5755355958965798) q[17];
cx q[16],q[17];
ry(0.9030739456512938) q[16];
ry(0.2411042253681637) q[17];
cx q[16],q[17];
ry(-1.5934713130744882) q[18];
ry(-1.1207083142337666) q[19];
cx q[18],q[19];
ry(1.5745884261304282) q[18];
ry(0.5348431839210104) q[19];
cx q[18],q[19];
ry(-3.06925770509108) q[1];
ry(0.20078820880716802) q[2];
cx q[1],q[2];
ry(-1.568922567529353) q[1];
ry(-1.5532930861813004) q[2];
cx q[1],q[2];
ry(1.2369153874022096) q[3];
ry(-1.5949930491573334) q[4];
cx q[3],q[4];
ry(-1.5509826402163842) q[3];
ry(3.0410394497588484) q[4];
cx q[3],q[4];
ry(0.6690933654549376) q[5];
ry(3.1257178066768603) q[6];
cx q[5],q[6];
ry(-1.7259709612278558) q[5];
ry(1.58618846112431) q[6];
cx q[5],q[6];
ry(-0.046999976966352726) q[7];
ry(-0.3448703469551572) q[8];
cx q[7],q[8];
ry(-0.00965576181847183) q[7];
ry(0.004255535307908836) q[8];
cx q[7],q[8];
ry(-0.06635959310435435) q[9];
ry(-2.8642791286886076) q[10];
cx q[9],q[10];
ry(0.00794634534462719) q[9];
ry(1.4269090458080598) q[10];
cx q[9],q[10];
ry(-3.141148633135569) q[11];
ry(-2.2763880464230497) q[12];
cx q[11],q[12];
ry(1.6719372462130204) q[11];
ry(-1.7339769498550397) q[12];
cx q[11],q[12];
ry(-2.8754749411431613) q[13];
ry(0.10485307256772548) q[14];
cx q[13],q[14];
ry(1.6972856128710845) q[13];
ry(-1.5686714332685066) q[14];
cx q[13],q[14];
ry(-1.9465488580235428) q[15];
ry(-3.12909970917318) q[16];
cx q[15],q[16];
ry(-1.7199679027507329) q[15];
ry(-1.605884505363683) q[16];
cx q[15],q[16];
ry(-1.5703087423446291) q[17];
ry(1.5937599879177318) q[18];
cx q[17],q[18];
ry(-1.0010223299219914) q[17];
ry(1.0531303728841777) q[18];
cx q[17],q[18];
ry(2.6963650959575127) q[0];
ry(-1.7624410001711468) q[1];
ry(2.8762153594053923) q[2];
ry(2.766483209028483) q[3];
ry(3.0517864646141675) q[4];
ry(1.3124637948392355) q[5];
ry(1.2854334543930537) q[6];
ry(-2.3432798729230218) q[7];
ry(-1.447510856639336) q[8];
ry(3.0023621631921453) q[9];
ry(1.0885802642165925) q[10];
ry(1.380571414791453) q[11];
ry(-0.2395871356501934) q[12];
ry(1.3722932772741214) q[13];
ry(-1.1175806044398051) q[14];
ry(-1.8011385012036536) q[15];
ry(-1.7053280540147289) q[16];
ry(2.926978461381231) q[17];
ry(-1.728067894841462) q[18];
ry(2.9290916901185713) q[19];