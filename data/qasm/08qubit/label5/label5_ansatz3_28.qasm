OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.7897057181883997) q[0];
rz(0.5131266182317633) q[0];
ry(-0.23887709070870056) q[1];
rz(-0.7529090578786627) q[1];
ry(-0.9218755322054992) q[2];
rz(2.7987617792872213) q[2];
ry(0.6266533204684583) q[3];
rz(-1.0073194302736113) q[3];
ry(-3.0493944729751794) q[4];
rz(2.012473721907697) q[4];
ry(1.3406279079740218) q[5];
rz(-2.8775947572735925) q[5];
ry(-0.45084892059157283) q[6];
rz(1.0732458780678558) q[6];
ry(0.6521015478240555) q[7];
rz(-1.996786722730078) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.968494649999869) q[0];
rz(1.7310324194092486) q[0];
ry(0.12209438716477372) q[1];
rz(0.9795753787913706) q[1];
ry(-2.833501750468077) q[2];
rz(1.3319590806544266) q[2];
ry(-1.0852292204416498) q[3];
rz(-2.3896266207139085) q[3];
ry(-2.0445746874850244) q[4];
rz(2.5847674189800216) q[4];
ry(0.43722043863593285) q[5];
rz(-0.7198400268197914) q[5];
ry(0.768413684321822) q[6];
rz(-0.07550814672980798) q[6];
ry(1.140039816259165) q[7];
rz(0.974592830534025) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.7997741672649328) q[0];
rz(-1.5799475852128129) q[0];
ry(2.269231988806796) q[1];
rz(-2.3957725832861296) q[1];
ry(0.8394986018046041) q[2];
rz(2.427199355934931) q[2];
ry(2.158063461406482) q[3];
rz(-0.21124340552204315) q[3];
ry(-0.7462364693371315) q[4];
rz(-1.992127730539723) q[4];
ry(-2.005061414189443) q[5];
rz(0.8062194506957269) q[5];
ry(-1.3203169037945504) q[6];
rz(-1.8866293468666602) q[6];
ry(1.725374029783623) q[7];
rz(-1.0294084838210436) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.4913076265662335) q[0];
rz(0.4494465825527367) q[0];
ry(-1.7848619591201027) q[1];
rz(-2.1295025641792287) q[1];
ry(-0.4834214452612109) q[2];
rz(-2.8351197046052525) q[2];
ry(1.187056238892941) q[3];
rz(-0.14998343672514913) q[3];
ry(-1.1758722477467611) q[4];
rz(-2.988122751745981) q[4];
ry(2.884142946267231) q[5];
rz(-0.3438693985387534) q[5];
ry(2.0693504576600708) q[6];
rz(-0.619652713912199) q[6];
ry(1.0603391699563602) q[7];
rz(-1.0940674196149716) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.0112221146720952) q[0];
rz(-1.737675968054229) q[0];
ry(1.132423781278237) q[1];
rz(1.114017402109151) q[1];
ry(1.1597879656673884) q[2];
rz(2.0612076671630764) q[2];
ry(-1.810332242109559) q[3];
rz(-3.092182872641361) q[3];
ry(0.6691383918779197) q[4];
rz(-0.08671646283638589) q[4];
ry(0.11138895208549471) q[5];
rz(-2.7612275090406375) q[5];
ry(1.323975096336623) q[6];
rz(-0.7878968321559876) q[6];
ry(1.0572711472748262) q[7];
rz(0.026709340480475036) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.9592369386555148) q[0];
rz(1.8180281504613705) q[0];
ry(0.9160846746414976) q[1];
rz(-1.4450770344181654) q[1];
ry(1.1343274542965842) q[2];
rz(0.8265080833498183) q[2];
ry(1.1069407235392466) q[3];
rz(-2.537573233288562) q[3];
ry(1.3047710955005856) q[4];
rz(1.8045824407512892) q[4];
ry(-1.389236972973178) q[5];
rz(1.3445059997225313) q[5];
ry(-0.31403268822781705) q[6];
rz(-2.396997883470432) q[6];
ry(0.9465259559444074) q[7];
rz(2.991559466881396) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.5320208355426543) q[0];
rz(-1.3536709841058157) q[0];
ry(1.7614864715278684) q[1];
rz(-0.08385311445587519) q[1];
ry(1.726042294947022) q[2];
rz(-0.8228396274477917) q[2];
ry(-2.638588920200691) q[3];
rz(0.815602154522228) q[3];
ry(2.1885164942494906) q[4];
rz(-2.416737239312893) q[4];
ry(0.11874051855101017) q[5];
rz(-1.611082474020637) q[5];
ry(-1.7563912065878986) q[6];
rz(-2.388775806750298) q[6];
ry(-0.8926217363496574) q[7];
rz(2.682961226199338) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.6504224710871) q[0];
rz(1.767000519546323) q[0];
ry(-1.817863913925697) q[1];
rz(0.36124563982288427) q[1];
ry(-2.5129679708088197) q[2];
rz(-3.018472633324335) q[2];
ry(2.3853095761514918) q[3];
rz(0.591932399972568) q[3];
ry(0.6533102015124292) q[4];
rz(-1.677256512210318) q[4];
ry(0.41514584787966896) q[5];
rz(-1.0556859991015095) q[5];
ry(1.7042571871920267) q[6];
rz(-2.2099725251299294) q[6];
ry(-2.352408864381465) q[7];
rz(-0.6811484489333499) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.009134400657219608) q[0];
rz(-0.3041990293770098) q[0];
ry(-1.291899859456577) q[1];
rz(-0.9798394203267186) q[1];
ry(1.185276329847351) q[2];
rz(1.4626443962831575) q[2];
ry(-2.683449176543831) q[3];
rz(2.076516557622797) q[3];
ry(-1.345274789476959) q[4];
rz(2.408171936633717) q[4];
ry(-2.8685212043484616) q[5];
rz(-3.021992346823605) q[5];
ry(0.35664156408618036) q[6];
rz(1.000651360661357) q[6];
ry(2.611322480946377) q[7];
rz(-3.0852225750704894) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.964640822750086) q[0];
rz(-0.9403912028405843) q[0];
ry(2.074241070899867) q[1];
rz(2.437419145973586) q[1];
ry(-2.244911030796077) q[2];
rz(2.573030098916436) q[2];
ry(-2.8794238527331513) q[3];
rz(-3.072401991718567) q[3];
ry(-3.0638867335954822) q[4];
rz(-2.837522706105962) q[4];
ry(2.2204722390008382) q[5];
rz(2.492250803820535) q[5];
ry(-0.018567345506028267) q[6];
rz(0.29014200670811346) q[6];
ry(-0.6271078111799699) q[7];
rz(-2.596875935721172) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.4840119173563906) q[0];
rz(-1.0381757614174796) q[0];
ry(-0.19915847959487984) q[1];
rz(-3.1081155083418732) q[1];
ry(-2.409968953073539) q[2];
rz(-1.8663539604887045) q[2];
ry(1.2679139116237756) q[3];
rz(-1.7660873297566089) q[3];
ry(-1.3282140348213807) q[4];
rz(-2.0287314681853514) q[4];
ry(-2.16801818409217) q[5];
rz(-0.9748141529421063) q[5];
ry(-2.5240463678394196) q[6];
rz(0.3419647477348384) q[6];
ry(-2.9417936547836714) q[7];
rz(-0.5914703326135121) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.9951949417111173) q[0];
rz(1.857278998301803) q[0];
ry(2.0417874549818573) q[1];
rz(-2.065910740765733) q[1];
ry(0.38774084730134406) q[2];
rz(2.109856407768861) q[2];
ry(1.200008399109608) q[3];
rz(0.8412923721813758) q[3];
ry(-1.1326307641092326) q[4];
rz(0.48085086687278716) q[4];
ry(-1.045200070861044) q[5];
rz(0.7482530332065086) q[5];
ry(-1.279342189542181) q[6];
rz(-2.9022686613074895) q[6];
ry(-0.6144271953030191) q[7];
rz(-1.4641069023219389) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.6831491600174093) q[0];
rz(-2.869409935340551) q[0];
ry(0.02449207964621011) q[1];
rz(2.6084799848189437) q[1];
ry(1.1596774454033039) q[2];
rz(0.020652679927484142) q[2];
ry(1.2738860280374347) q[3];
rz(3.0970318677920274) q[3];
ry(0.6014084745761014) q[4];
rz(-0.5974546805326453) q[4];
ry(2.884519075439874) q[5];
rz(-0.6160281927168433) q[5];
ry(-2.42624731889501) q[6];
rz(-2.7435042808885295) q[6];
ry(-1.0196414465775872) q[7];
rz(2.007499166263348) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.6844567031943729) q[0];
rz(0.10156344752175558) q[0];
ry(-0.9391669054120053) q[1];
rz(-0.4280048106341887) q[1];
ry(0.5341217058728968) q[2];
rz(-1.2160678482909293) q[2];
ry(-0.9126560610056481) q[3];
rz(3.121000057496729) q[3];
ry(1.2578303573390874) q[4];
rz(-1.1791669518594095) q[4];
ry(-0.2549987483795375) q[5];
rz(-1.2786942770981575) q[5];
ry(-2.364922746971936) q[6];
rz(3.016142677476508) q[6];
ry(2.2915391678092396) q[7];
rz(-0.966995294400087) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.051592438381486) q[0];
rz(-1.0093413451953612) q[0];
ry(-0.43684196635293393) q[1];
rz(0.3863794552953683) q[1];
ry(-0.1425750693083403) q[2];
rz(0.6721659518754484) q[2];
ry(1.7916550299723726) q[3];
rz(-2.367956559398668) q[3];
ry(-1.6206301197319268) q[4];
rz(1.7096664515830753) q[4];
ry(-2.2275399687379935) q[5];
rz(-0.9422404765067356) q[5];
ry(-0.6227192577129593) q[6];
rz(-1.974557517937465) q[6];
ry(-1.2057206077679334) q[7];
rz(-0.8533051563458125) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.250080460053548) q[0];
rz(0.2516897486079613) q[0];
ry(-2.5109988610361) q[1];
rz(2.604749822874715) q[1];
ry(-2.235229571334505) q[2];
rz(2.371527307180152) q[2];
ry(0.7649585803110892) q[3];
rz(-0.789434333347426) q[3];
ry(1.6444354879168555) q[4];
rz(-0.23902894753311182) q[4];
ry(-1.7150131717921862) q[5];
rz(2.8382111975948154) q[5];
ry(-2.975095923119219) q[6];
rz(-2.842376390383024) q[6];
ry(-1.6108795027516136) q[7];
rz(-0.07901311201607593) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.8156975658638981) q[0];
rz(-1.170253379453471) q[0];
ry(0.4570561937407408) q[1];
rz(2.251727896609898) q[1];
ry(1.9829592519450565) q[2];
rz(-2.1419995431039283) q[2];
ry(0.9805492794736251) q[3];
rz(2.6966687282664044) q[3];
ry(2.017593730329708) q[4];
rz(-2.8960473557257815) q[4];
ry(2.3209299908543306) q[5];
rz(3.0103770517035993) q[5];
ry(1.3554216076551349) q[6];
rz(-2.7509062977730325) q[6];
ry(2.160065638002362) q[7];
rz(-1.7354917387713673) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.450278998918028) q[0];
rz(-2.4988079705032753) q[0];
ry(0.5535375928070589) q[1];
rz(-2.7714070460710087) q[1];
ry(0.7283541859044768) q[2];
rz(1.716141454848906) q[2];
ry(-2.788405164847599) q[3];
rz(-1.5697331946878377) q[3];
ry(-1.9784387559531142) q[4];
rz(-0.9993539097856593) q[4];
ry(1.2251526306226772) q[5];
rz(-1.3007491103683044) q[5];
ry(-0.12932971943172575) q[6];
rz(2.645752893776271) q[6];
ry(2.4331637669346016) q[7];
rz(-2.7998408008920035) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.3139513774824572) q[0];
rz(-1.789621702916161) q[0];
ry(0.9947069112947755) q[1];
rz(1.027244484984032) q[1];
ry(-2.5189217370652335) q[2];
rz(-2.9745089319174367) q[2];
ry(-2.7111633137379836) q[3];
rz(2.4409811066589726) q[3];
ry(-0.25986737904562585) q[4];
rz(-2.565440202490213) q[4];
ry(1.3523417378890388) q[5];
rz(-2.7963702510437325) q[5];
ry(-1.9911126337197829) q[6];
rz(-0.9802498537238425) q[6];
ry(-1.6388772439123658) q[7];
rz(-2.4220914683236154) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.8624807078324765) q[0];
rz(1.2623180654835355) q[0];
ry(0.5837506478594615) q[1];
rz(-0.31202829799668746) q[1];
ry(-1.7518468439561512) q[2];
rz(0.6463937556905348) q[2];
ry(1.5451824240856975) q[3];
rz(-0.2213153167338333) q[3];
ry(-1.6909066419840864) q[4];
rz(-2.5228698228014657) q[4];
ry(-1.9430701770527101) q[5];
rz(-0.6903883947146879) q[5];
ry(-0.2788599937511087) q[6];
rz(3.044592976214484) q[6];
ry(2.223363309268817) q[7];
rz(-3.1403891427185293) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.8495574214954138) q[0];
rz(2.0353841582813277) q[0];
ry(1.575347492106182) q[1];
rz(0.5292973431793175) q[1];
ry(-2.80296095443235) q[2];
rz(2.178518730078414) q[2];
ry(0.37831612222279176) q[3];
rz(1.9109427531071752) q[3];
ry(-2.5946379759726983) q[4];
rz(-1.0100708672256546) q[4];
ry(2.9799241810137684) q[5];
rz(1.2087231765515254) q[5];
ry(-2.155373860637553) q[6];
rz(2.735522878868138) q[6];
ry(2.4832856143397684) q[7];
rz(-2.223009339115028) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.2670533943490545) q[0];
rz(3.0888091792364243) q[0];
ry(-2.3145228765188093) q[1];
rz(2.660637797758951) q[1];
ry(-0.8781439703770858) q[2];
rz(-2.938607001528917) q[2];
ry(2.9302237278197505) q[3];
rz(1.3775414184974875) q[3];
ry(2.453726104670479) q[4];
rz(2.3078683414959764) q[4];
ry(-2.6969431353171784) q[5];
rz(-1.2428223660859616) q[5];
ry(0.08625367332353487) q[6];
rz(-1.2459990207626752) q[6];
ry(-2.8482019498697415) q[7];
rz(2.2112504235466917) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.291507480548045) q[0];
rz(-1.851066690244748) q[0];
ry(-1.1538104775304527) q[1];
rz(-2.0458292139752166) q[1];
ry(-2.526698972314109) q[2];
rz(-1.690583203837859) q[2];
ry(-3.024371368918827) q[3];
rz(-1.1319635866215276) q[3];
ry(0.7841866186925454) q[4];
rz(2.8067509858529167) q[4];
ry(0.7266110204666996) q[5];
rz(-0.7956355541593795) q[5];
ry(-1.8745339581147435) q[6];
rz(-1.4302486598586373) q[6];
ry(-1.1474246303022575) q[7];
rz(-0.7325156242959735) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.39297246336778713) q[0];
rz(1.5340681196265398) q[0];
ry(-1.578782471315204) q[1];
rz(0.7411453934586003) q[1];
ry(-0.35154087540248735) q[2];
rz(2.0323521068397326) q[2];
ry(1.1697195046506046) q[3];
rz(-0.4425106294817329) q[3];
ry(2.808583512515143) q[4];
rz(2.464451532802565) q[4];
ry(-0.3816398561363018) q[5];
rz(-1.130915324966434) q[5];
ry(1.9464427983792947) q[6];
rz(-1.5355127817771572) q[6];
ry(1.5605707123296888) q[7];
rz(0.9193456896180026) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.2797561162382044) q[0];
rz(3.039396901827088) q[0];
ry(-0.5359103726626346) q[1];
rz(-0.6529045568125175) q[1];
ry(1.5546657845144853) q[2];
rz(3.0753876415729424) q[2];
ry(0.6440293522843386) q[3];
rz(-1.873392492991992) q[3];
ry(-2.6750026190944745) q[4];
rz(2.2356013519199136) q[4];
ry(-2.4874147643157083) q[5];
rz(-2.768660497739211) q[5];
ry(2.7527374804190257) q[6];
rz(-2.11749203338046) q[6];
ry(1.504602593310165) q[7];
rz(3.110398354554672) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.8583305041014118) q[0];
rz(-0.6686099384562948) q[0];
ry(-2.7019351515204137) q[1];
rz(-0.09197141307946426) q[1];
ry(-1.287682332384829) q[2];
rz(1.7279064521913339) q[2];
ry(1.5850032742969633) q[3];
rz(-0.7620451267647393) q[3];
ry(2.604762643376382) q[4];
rz(1.028692328557975) q[4];
ry(0.8909439783872531) q[5];
rz(-2.2895836331789217) q[5];
ry(-0.39201330620910585) q[6];
rz(-1.0793879624066138) q[6];
ry(1.4419159593724) q[7];
rz(-0.24465334726881768) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.1361839652170969) q[0];
rz(2.278574189251737) q[0];
ry(-1.3494024975216834) q[1];
rz(-2.6403302022934425) q[1];
ry(0.8537479963031824) q[2];
rz(-1.4064480762463296) q[2];
ry(0.5978614192848255) q[3];
rz(0.6836617364814267) q[3];
ry(-2.2002380858298487) q[4];
rz(-0.6541274134291513) q[4];
ry(-2.4848539588888827) q[5];
rz(-1.975283044500757) q[5];
ry(2.570431026031218) q[6];
rz(2.194685942627795) q[6];
ry(2.008644674100475) q[7];
rz(-2.9011628659929967) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.438185633128231) q[0];
rz(0.49324857221556684) q[0];
ry(1.1514295846496716) q[1];
rz(1.934749408083609) q[1];
ry(-1.5501509530740534) q[2];
rz(-1.9734085420719092) q[2];
ry(-2.5520100736381868) q[3];
rz(2.2199968285140086) q[3];
ry(2.968395030459219) q[4];
rz(0.13744496973314568) q[4];
ry(-0.1519278113675462) q[5];
rz(1.3870831744927399) q[5];
ry(-0.5173876640020264) q[6];
rz(-0.8958113535635346) q[6];
ry(2.4391374585517402) q[7];
rz(-1.6745840668857719) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.492681178707946) q[0];
rz(-2.0345507334633917) q[0];
ry(-0.48430535888506565) q[1];
rz(0.5192824760263024) q[1];
ry(1.7361506640014481) q[2];
rz(-1.2856347248455258) q[2];
ry(1.6486217960328649) q[3];
rz(-1.7205699652171864) q[3];
ry(1.2400106243125188) q[4];
rz(0.23306187653788246) q[4];
ry(-1.5383975047987626) q[5];
rz(2.8009945103176346) q[5];
ry(0.5814305035311556) q[6];
rz(0.9212600127910819) q[6];
ry(-2.46775681044921) q[7];
rz(1.4149347836167503) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.7561067840674527) q[0];
rz(2.850964393759086) q[0];
ry(-0.7759277637084773) q[1];
rz(0.8854917677331962) q[1];
ry(1.2888215420845508) q[2];
rz(-0.9556830243908053) q[2];
ry(-3.02790252845533) q[3];
rz(-1.566665938941454) q[3];
ry(2.330063505487561) q[4];
rz(-0.46189181844154803) q[4];
ry(0.8756142891182596) q[5];
rz(2.536541100271755) q[5];
ry(1.4831831251724186) q[6];
rz(3.0540898908031697) q[6];
ry(-2.0195896312229173) q[7];
rz(-0.3948952482047785) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.782207373954356) q[0];
rz(-1.6461907736462684) q[0];
ry(2.614758820702777) q[1];
rz(-0.9574092611352675) q[1];
ry(-2.0888624760685275) q[2];
rz(2.984788191740145) q[2];
ry(0.7158622108117516) q[3];
rz(2.6313332641038802) q[3];
ry(-2.4054861208196554) q[4];
rz(-2.1045011120389274) q[4];
ry(-1.2530175560614145) q[5];
rz(0.3087250957858176) q[5];
ry(0.5446954411655608) q[6];
rz(2.210443714520904) q[6];
ry(0.9725334518821569) q[7];
rz(2.644537710871714) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.8577821498457379) q[0];
rz(0.5689770180766126) q[0];
ry(0.3457368830970093) q[1];
rz(0.7169438868065641) q[1];
ry(1.0518120554405206) q[2];
rz(-1.7057620962038442) q[2];
ry(-2.6008975433457353) q[3];
rz(-1.1122902960750631) q[3];
ry(-0.6951296109870038) q[4];
rz(-0.0028277040582080915) q[4];
ry(-0.6324060846978163) q[5];
rz(-2.977485447317551) q[5];
ry(1.9191686686570606) q[6];
rz(1.6594096307277146) q[6];
ry(-1.218183617702473) q[7];
rz(1.1542473903881207) q[7];