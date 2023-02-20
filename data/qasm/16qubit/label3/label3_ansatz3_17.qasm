OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-1.1880228507946708) q[0];
rz(-1.5243050522897008) q[0];
ry(1.934527399478486) q[1];
rz(-0.8423428136920019) q[1];
ry(2.971487727099704) q[2];
rz(-0.27144020018284637) q[2];
ry(2.8993729251524947) q[3];
rz(1.550255501854772) q[3];
ry(-0.27733916883239385) q[4];
rz(-0.5639135577886972) q[4];
ry(3.0665135924415137) q[5];
rz(0.2689649304869324) q[5];
ry(0.7916275843923417) q[6];
rz(-1.0976667137549745) q[6];
ry(0.0436556009519089) q[7];
rz(-2.9226626258493034) q[7];
ry(-1.577421748479811) q[8];
rz(-1.5421805337776981) q[8];
ry(-2.2247288647520165) q[9];
rz(-1.753915373966124) q[9];
ry(3.1396319980218914) q[10];
rz(-1.216599459119907) q[10];
ry(-1.6369190981752035) q[11];
rz(1.5588418464510925) q[11];
ry(-0.14171570795289964) q[12];
rz(-0.07614591175532581) q[12];
ry(3.1409085484913013) q[13];
rz(0.3824047295422375) q[13];
ry(-1.713679192549722) q[14];
rz(2.406557386423622) q[14];
ry(1.8890875086997523) q[15];
rz(1.9406344580217998) q[15];
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
ry(-1.94910309912962) q[0];
rz(0.5974555462553106) q[0];
ry(0.9030356344665672) q[1];
rz(-1.485101572579534) q[1];
ry(3.0687661627665603) q[2];
rz(-1.3144096533892633) q[2];
ry(2.8622111389632003) q[3];
rz(-1.2820137898938329) q[3];
ry(3.1406676728359417) q[4];
rz(0.7458493544669617) q[4];
ry(-0.09074630000792805) q[5];
rz(-1.7378460136438105) q[5];
ry(-0.011675697449692104) q[6];
rz(1.4938159732338478) q[6];
ry(0.07693481491918057) q[7];
rz(1.2766419471248656) q[7];
ry(2.88220740731888) q[8];
rz(2.70784498016026) q[8];
ry(3.1411664263642716) q[9];
rz(-1.5202206904939315) q[9];
ry(-0.39655630860611946) q[10];
rz(0.6770099470292019) q[10];
ry(-1.6522301087770637) q[11];
rz(-3.1274839310565716) q[11];
ry(1.5486208700145747) q[12];
rz(2.002309634806516) q[12];
ry(0.08159649447945115) q[13];
rz(0.4062165650855114) q[13];
ry(-2.5991759466202353) q[14];
rz(2.928159379275937) q[14];
ry(-1.6744529392342564) q[15];
rz(-1.6680473847798796) q[15];
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
ry(-0.7767723676328189) q[0];
rz(2.8866941368883463) q[0];
ry(2.2584199630754105) q[1];
rz(-1.2168003864520454) q[1];
ry(1.3817884723589617) q[2];
rz(2.945889530603107) q[2];
ry(0.07117282639151057) q[3];
rz(0.501538457215731) q[3];
ry(-1.0778678245674609) q[4];
rz(-2.83273333045738) q[4];
ry(-3.0840499156585977) q[5];
rz(2.9502661309048626) q[5];
ry(-0.8037000549456748) q[6];
rz(-0.7337938031918981) q[6];
ry(3.1266988267924365) q[7];
rz(-1.9702800634550015) q[7];
ry(0.030095508583629457) q[8];
rz(2.041016042825608) q[8];
ry(-1.6117470187028808) q[9];
rz(-0.562298325640623) q[9];
ry(-3.1393690815300404) q[10];
rz(1.8986416158932335) q[10];
ry(-0.33181901338188435) q[11];
rz(1.645502619635156) q[11];
ry(-0.6757618587088068) q[12];
rz(1.3200695292650888) q[12];
ry(3.1403246906584963) q[13];
rz(-0.687704162340216) q[13];
ry(-2.9433956133264902) q[14];
rz(1.2851244157817845) q[14];
ry(1.365857096778019) q[15];
rz(-0.36146969108261645) q[15];
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
ry(-1.6747617384898918) q[0];
rz(-2.4158168152452744) q[0];
ry(-0.640985764333438) q[1];
rz(2.257436526183551) q[1];
ry(-0.016014806094458045) q[2];
rz(-2.613572698729976) q[2];
ry(1.0174560615944392) q[3];
rz(-1.1463894329704578) q[3];
ry(-1.0843597400530538) q[4];
rz(0.0015830053624990017) q[4];
ry(3.115869074917736) q[5];
rz(1.3584736764081704) q[5];
ry(0.0012639169046615155) q[6];
rz(2.0949741780306805) q[6];
ry(-2.8596103595633293) q[7];
rz(-2.2980284538967575) q[7];
ry(1.5731073699542852) q[8];
rz(-0.49573315802617623) q[8];
ry(3.1048568987085843) q[9];
rz(-1.847939549457232) q[9];
ry(1.574051516454646) q[10];
rz(-1.0113229847457672) q[10];
ry(-1.4545490584426295) q[11];
rz(-1.6069300222800986) q[11];
ry(-3.1140167793054174) q[12];
rz(1.44927431367329) q[12];
ry(1.5647221560239357) q[13];
rz(2.273979205985863) q[13];
ry(-1.7038038248729324) q[14];
rz(3.1194808763869926) q[14];
ry(0.30307462861848844) q[15];
rz(-2.0582346563046148) q[15];
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
ry(2.3243491382583934) q[0];
rz(-0.846021480585434) q[0];
ry(1.5862236895548865) q[1];
rz(-0.5726426315781052) q[1];
ry(3.13464053782766) q[2];
rz(2.931445218482982) q[2];
ry(0.02061494433554536) q[3];
rz(2.515623054638716) q[3];
ry(-2.1180762123912906) q[4];
rz(3.140484016443866) q[4];
ry(-3.139104407332291) q[5];
rz(-0.3756011631487545) q[5];
ry(-1.580828451324737) q[6];
rz(0.8609668975806762) q[6];
ry(0.0011380587987839105) q[7];
rz(0.7097597564461243) q[7];
ry(-3.140386220431977) q[8];
rz(2.328847253426755) q[8];
ry(0.00468106782214699) q[9];
rz(2.991432740954773) q[9];
ry(-0.012323281440704333) q[10];
rz(-1.3955007553124892) q[10];
ry(-1.658994338566675) q[11];
rz(-0.8720363244165125) q[11];
ry(-0.008715132372609347) q[12];
rz(2.811888686126004) q[12];
ry(-3.1079022398658456) q[13];
rz(2.350143756608016) q[13];
ry(0.37327507552919814) q[14];
rz(0.012401486881467397) q[14];
ry(1.5119992194562735) q[15];
rz(0.136466999920696) q[15];
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
ry(-0.018832846744651993) q[0];
rz(-1.1941649864095825) q[0];
ry(-1.607741400115791) q[1];
rz(1.5033259601376285) q[1];
ry(-0.18249026682913347) q[2];
rz(-2.9877265403791764) q[2];
ry(-0.16241799533921286) q[3];
rz(-1.602138084854638) q[3];
ry(-1.5686635385001249) q[4];
rz(1.6012692331313965) q[4];
ry(0.2149486569144452) q[5];
rz(-0.23068117997429893) q[5];
ry(0.24863358983231176) q[6];
rz(-1.4452399559807843) q[6];
ry(3.140464601380607) q[7];
rz(-0.032619170036139095) q[7];
ry(2.8412970492841487) q[8];
rz(-0.7424910522033281) q[8];
ry(-1.5695976568813954) q[9];
rz(0.057423762179449694) q[9];
ry(0.0006865891586013854) q[10];
rz(1.0685753057026186) q[10];
ry(0.04394275795326763) q[11];
rz(0.8845170761965379) q[11];
ry(3.0283913038856247) q[12];
rz(1.3800675717463484) q[12];
ry(3.109488241425777) q[13];
rz(-3.128713906774971) q[13];
ry(-1.4983315882175148) q[14];
rz(2.6975257502329173) q[14];
ry(-0.3893387103769515) q[15];
rz(-0.07887209169163435) q[15];
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
ry(-0.21695391472884062) q[0];
rz(-0.5675422066997771) q[0];
ry(2.03329710229082) q[1];
rz(-0.00439795347532741) q[1];
ry(-1.3285127162504526) q[2];
rz(-3.132790987504886) q[2];
ry(0.14288245175658998) q[3];
rz(-1.4335070890630033) q[3];
ry(3.138911296005879) q[4];
rz(-0.11729371362010799) q[4];
ry(-3.1231232160273144) q[5];
rz(2.740129187725862) q[5];
ry(1.525055266443423) q[6];
rz(-0.37929520999005584) q[6];
ry(-3.1379188261856945) q[7];
rz(2.2212530658390666) q[7];
ry(-0.006808424440520049) q[8];
rz(-1.3576507605780703) q[8];
ry(-2.9239700135002322) q[9];
rz(-2.001990306762684) q[9];
ry(1.0193204200755128) q[10];
rz(2.079957677159486) q[10];
ry(-1.6933248114888793) q[11];
rz(1.2705654266222508) q[11];
ry(-1.550955076840169) q[12];
rz(3.140948469588867) q[12];
ry(-1.5694986852283357) q[13];
rz(-0.027275058929280505) q[13];
ry(-0.04165830897832123) q[14];
rz(2.3641616789966347) q[14];
ry(1.438259668071006) q[15];
rz(-0.09009545357416114) q[15];
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
ry(-1.3086428224315882) q[0];
rz(-1.143736943675342) q[0];
ry(2.0242634044260193) q[1];
rz(1.9826728193569148) q[1];
ry(-1.5194066737762955) q[2];
rz(0.19837896272750286) q[2];
ry(1.3409727695682587) q[3];
rz(1.4520287399021283) q[3];
ry(3.124655304749857) q[4];
rz(-0.7406316878971444) q[4];
ry(0.20797111907186494) q[5];
rz(-1.4061569416359525) q[5];
ry(-0.19496766147092792) q[6];
rz(1.7848884563759562) q[6];
ry(0.003528107345494376) q[7];
rz(-0.5115372042673444) q[7];
ry(-3.1386281939215093) q[8];
rz(-0.6907826978649553) q[8];
ry(-0.03840925184329756) q[9];
rz(2.793240278965994) q[9];
ry(3.1415696530085717) q[10];
rz(-0.1900307434774007) q[10];
ry(0.0011803278926231903) q[11];
rz(1.9038990712540442) q[11];
ry(-1.6117547416555187) q[12];
rz(3.0992411195027145) q[12];
ry(-2.0278737551524255) q[13];
rz(-0.11966886533307176) q[13];
ry(1.8188104439361863) q[14];
rz(-0.38651445663731376) q[14];
ry(0.2419680020728745) q[15];
rz(-1.7953838995944826) q[15];
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
ry(0.053606109713791206) q[0];
rz(-1.8302687004067817) q[0];
ry(-0.43398667030265276) q[1];
rz(1.2467501638009535) q[1];
ry(-1.56202964794097) q[2];
rz(-0.6108787353022657) q[2];
ry(1.4793228745180533) q[3];
rz(2.5115151932286466) q[3];
ry(1.5654392905960763) q[4];
rz(2.76273395240618) q[4];
ry(-3.1342353252882336) q[5];
rz(-1.6482765436768436) q[5];
ry(0.9929209321043336) q[6];
rz(-1.2625878849379288) q[6];
ry(-1.5685918792587517) q[7];
rz(2.5949702589135053) q[7];
ry(3.1352565130622922) q[8];
rz(0.7987902306986596) q[8];
ry(0.06436465749608457) q[9];
rz(-2.0814561948063126) q[9];
ry(2.642145425066099) q[10];
rz(-0.7215827599855587) q[10];
ry(-0.31336125073930887) q[11];
rz(-1.8115142492854677) q[11];
ry(-0.0019447946640477644) q[12];
rz(2.3907961543539815) q[12];
ry(0.011533255839403012) q[13];
rz(1.685186820869398) q[13];
ry(-3.122131182859217) q[14];
rz(-0.38836470414607405) q[14];
ry(-0.013576329130571276) q[15];
rz(2.2200745577363588) q[15];
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
ry(2.308795033873045) q[0];
rz(-1.5498367126929278) q[0];
ry(-1.5963722201201254) q[1];
rz(0.02750552932601626) q[1];
ry(-1.5716157101303694) q[2];
rz(1.5672782395748122) q[2];
ry(1.741809547198331) q[3];
rz(1.6230246007863314) q[3];
ry(-0.0060782171753892555) q[4];
rz(-1.197731135318941) q[4];
ry(1.5706689730327739) q[5];
rz(0.5341282864599556) q[5];
ry(3.1407134043116343) q[6];
rz(2.167722226105967) q[6];
ry(-0.00937502189695838) q[7];
rz(2.51214296740041) q[7];
ry(0.0011104538485651404) q[8];
rz(0.5697006569228981) q[8];
ry(-0.04861841989600988) q[9];
rz(1.0478167582813178) q[9];
ry(1.5732336594362886) q[10];
rz(-3.139755567791461) q[10];
ry(-3.139970675212642) q[11];
rz(0.0066595379297208586) q[11];
ry(-3.129298631655551) q[12];
rz(0.78277070871674) q[12];
ry(1.5723783936622344) q[13];
rz(-2.871345520144984) q[13];
ry(1.3256722162956502) q[14];
rz(-0.7593373150610335) q[14];
ry(2.1140095403823516) q[15];
rz(0.7290104913517239) q[15];
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
ry(-1.556704814659509) q[0];
rz(3.1299336835995146) q[0];
ry(1.5589393945116354) q[1];
rz(1.454331217586032) q[1];
ry(-1.5678150489606935) q[2];
rz(-1.668871782201733) q[2];
ry(-1.5480916931207016) q[3];
rz(0.0029286582491820927) q[3];
ry(1.5590817874019232) q[4];
rz(0.015635252733410522) q[4];
ry(3.1404232971613344) q[5];
rz(-2.313113699659954) q[5];
ry(-0.5560999628355556) q[6];
rz(-0.8798042106463396) q[6];
ry(0.05693616337914017) q[7];
rz(0.18670556337191524) q[7];
ry(3.139212983662298) q[8];
rz(-0.8942107209812674) q[8];
ry(3.102468440408892) q[9];
rz(-2.1902162797830185) q[9];
ry(1.1966964981947088) q[10];
rz(1.5754082203903275) q[10];
ry(-3.141430751156456) q[11];
rz(1.5739649535193534) q[11];
ry(1.5865097724091333) q[12];
rz(0.25789539502030023) q[12];
ry(3.1402934750030664) q[13];
rz(2.261548236450505) q[13];
ry(0.034783807849699144) q[14];
rz(-2.1456584712821067) q[14];
ry(2.0415751935582334) q[15];
rz(3.134344923942878) q[15];
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
ry(1.5822073685049398) q[0];
rz(-1.7802248908601481) q[0];
ry(1.5708380508302957) q[1];
rz(2.712188479091404) q[1];
ry(-1.5705962831192002) q[2];
rz(1.0954394208118208) q[2];
ry(1.570514463636483) q[3];
rz(3.1410883362139335) q[3];
ry(-0.10201953714076417) q[4];
rz(1.5635036713425943) q[4];
ry(-3.1399977706556155) q[5];
rz(3.058603604519618) q[5];
ry(0.0019157808518696347) q[6];
rz(1.282530573340174) q[6];
ry(-0.003744665680681358) q[7];
rz(3.0334966743405287) q[7];
ry(-0.01414178033856519) q[8];
rz(-3.0007498567807684) q[8];
ry(0.0006933775462880705) q[9];
rz(1.7287988702076929) q[9];
ry(-1.5670042904256551) q[10];
rz(-1.6831662959073699) q[10];
ry(-0.02007620500542817) q[11];
rz(0.028440858724762205) q[11];
ry(-1.5562303656883614) q[12];
rz(1.5159437467323293) q[12];
ry(0.008908177602369172) q[13];
rz(-2.8302449779676504) q[13];
ry(-0.004998156225880734) q[14];
rz(0.9136575721927125) q[14];
ry(2.725020126790464) q[15];
rz(-0.9868945062941972) q[15];
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
ry(1.9759651075591098) q[0];
rz(0.32228109947828015) q[0];
ry(2.773159654610416) q[1];
rz(1.1840808702631813) q[1];
ry(1.9121731939990223) q[2];
rz(1.3985818295428143) q[2];
ry(1.6358723025388076) q[3];
rz(-2.9505925403582047) q[3];
ry(3.0350840876773586) q[4];
rz(-1.9916834475567988) q[4];
ry(-0.11761208471828791) q[5];
rz(-1.7800382078679764) q[5];
ry(0.8877655233567356) q[6];
rz(0.015663844773213257) q[6];
ry(3.0787078870019857) q[7];
rz(2.2605834311402226) q[7];
ry(3.111086397094789) q[8];
rz(1.0804939019924253) q[8];
ry(-2.8560433419063105) q[9];
rz(1.7925320106231422) q[9];
ry(-3.0488863375421102) q[10];
rz(-0.1603367957556873) q[10];
ry(0.004707098760033722) q[11];
rz(3.1322151281921413) q[11];
ry(1.579533462797594) q[12];
rz(-2.540040895282492) q[12];
ry(3.1384106806592986) q[13];
rz(-3.028148686634628) q[13];
ry(1.6392991336137994) q[14];
rz(-2.2490622635476174) q[14];
ry(0.7345265987000662) q[15];
rz(-3.085537524092027) q[15];
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
ry(3.1161302147708385) q[0];
rz(-2.82312445989769) q[0];
ry(-0.009965044428175685) q[1];
rz(-0.17388178740273116) q[1];
ry(-1.5869743684257136) q[2];
rz(0.00753065920853313) q[2];
ry(-0.00019887413545482957) q[3];
rz(-1.7840933965258916) q[3];
ry(-3.133027746591062) q[4];
rz(-0.6656068316968833) q[4];
ry(0.19000178391990818) q[5];
rz(-1.0469403942286037) q[5];
ry(3.1367169809623197) q[6];
rz(-0.6104053846603027) q[6];
ry(0.0034321698949568713) q[7];
rz(-0.7927701620081054) q[7];
ry(0.0018730614572213256) q[8];
rz(2.4866606830457374) q[8];
ry(-0.03394436734698145) q[9];
rz(2.7350590881068975) q[9];
ry(1.9724844314406704) q[10];
rz(3.0775512180143365) q[10];
ry(0.02022820684998905) q[11];
rz(1.4513006713731245) q[11];
ry(-1.6775901503414188) q[12];
rz(-0.5919171593588909) q[12];
ry(-0.5037514906929138) q[13];
rz(-2.943084806614639) q[13];
ry(1.5301856027311125) q[14];
rz(1.571739034746397) q[14];
ry(-2.873764218603361) q[15];
rz(-1.40678047483203) q[15];
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
ry(-1.5205712472042778) q[0];
rz(-0.8082007871052843) q[0];
ry(1.2119053514356422) q[1];
rz(0.6648416078336662) q[1];
ry(-1.8185745226916215) q[2];
rz(-0.003045855614573561) q[2];
ry(0.00030937158193822256) q[3];
rz(0.190680278292362) q[3];
ry(3.121477783592654) q[4];
rz(2.60349556699846) q[4];
ry(-1.464529725836939) q[5];
rz(0.30724339529283007) q[5];
ry(2.3756524310570697) q[6];
rz(-0.8396717493003322) q[6];
ry(-0.3899962068766191) q[7];
rz(-3.1340613125341155) q[7];
ry(3.134486319051168) q[8];
rz(2.826591079202732) q[8];
ry(-1.4931059416738766) q[9];
rz(2.603919598817263) q[9];
ry(-0.7191121493208011) q[10];
rz(3.1323644002496165) q[10];
ry(0.012926545207165852) q[11];
rz(0.4484608512922952) q[11];
ry(3.1184003420411415) q[12];
rz(-1.5357783374111902) q[12];
ry(0.003127291867402517) q[13];
rz(-0.7295225993922213) q[13];
ry(-3.111689690098343) q[14];
rz(3.087911669241514) q[14];
ry(-3.135932335890481) q[15];
rz(-1.7595908417473503) q[15];
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
ry(3.136984423593721) q[0];
rz(-0.8136290937274079) q[0];
ry(-1.5311161570732912) q[1];
rz(3.06946260640542) q[1];
ry(2.7351743395335024) q[2];
rz(-0.02377072939354214) q[2];
ry(-3.125930870814366) q[3];
rz(-1.395832355152578) q[3];
ry(-0.0012819473031848716) q[4];
rz(-1.2700140052199558) q[4];
ry(-3.0247186100944345) q[5];
rz(0.9032707626264167) q[5];
ry(-3.1387199482750217) q[6];
rz(-1.1297825739746452) q[6];
ry(-1.5693987676497292) q[7];
rz(0.00868514732502401) q[7];
ry(-0.08372343016691364) q[8];
rz(-2.2175843165295523) q[8];
ry(0.01684712803334038) q[9];
rz(-2.984092117704093) q[9];
ry(-0.38710509344111693) q[10];
rz(0.036255357747043256) q[10];
ry(0.005408157593750573) q[11];
rz(3.0378154409351747) q[11];
ry(-0.4376619258555045) q[12];
rz(1.6813335070809945) q[12];
ry(-2.056636714342053) q[13];
rz(0.8934125167329894) q[13];
ry(0.5670613774837275) q[14];
rz(-3.1094085795076047) q[14];
ry(0.20884844588312626) q[15];
rz(-1.5873906286614885) q[15];
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
ry(-1.5750488579229645) q[0];
rz(-1.5734288858001824) q[0];
ry(-1.8266658791445676) q[1];
rz(-1.624903493763072) q[1];
ry(2.2625347735411028) q[2];
rz(-0.008748627214010617) q[2];
ry(3.066558465094076) q[3];
rz(0.36430490816854993) q[3];
ry(1.568073954209689) q[4];
rz(-0.16477687689753573) q[4];
ry(-3.139611475375539) q[5];
rz(-2.6533755102949805) q[5];
ry(2.981490358756935) q[6];
rz(0.005279770805597827) q[6];
ry(2.9573863914886562) q[7];
rz(0.9460667756269548) q[7];
ry(-3.1275803623102614) q[8];
rz(0.27608965945420305) q[8];
ry(-0.5701016628903268) q[9];
rz(0.0016908832225110972) q[9];
ry(2.4507786949144235) q[10];
rz(-3.117824249192343) q[10];
ry(2.90322206542501) q[11];
rz(-2.0067398725604395) q[11];
ry(-0.0054829054692960705) q[12];
rz(2.48053531726052) q[12];
ry(3.1395914476879803) q[13];
rz(1.1376306863046388) q[13];
ry(-1.6377910079036493) q[14];
rz(0.045919520676066465) q[14];
ry(-0.01009662837005898) q[15];
rz(1.1969726153414382) q[15];
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
ry(-1.594345335450169) q[0];
rz(1.2094437448738105) q[0];
ry(-1.1093174330596378) q[1];
rz(-1.5013026302911792) q[1];
ry(-1.5716913557435461) q[2];
rz(-0.7586805696442527) q[2];
ry(-1.5622698745722154) q[3];
rz(-1.5906367182379286) q[3];
ry(-0.2696360840379599) q[4];
rz(-1.1430745166009573) q[4];
ry(3.1411400534995737) q[5];
rz(-2.52798010814064) q[5];
ry(-1.5980671314072268) q[6];
rz(0.08141427561194446) q[6];
ry(-0.0003689752686470982) q[7];
rz(-0.8945581242421445) q[7];
ry(2.842543095881972) q[8];
rz(-1.3128708124645898) q[8];
ry(-1.5719958937325966) q[9];
rz(3.1404708835721005) q[9];
ry(1.5874344523680737) q[10];
rz(0.10870307743876141) q[10];
ry(3.1404816221403147) q[11];
rz(-1.4057866335223166) q[11];
ry(1.4611992462647756) q[12];
rz(2.506929206886617) q[12];
ry(1.528796620933357) q[13];
rz(2.9485468981204153) q[13];
ry(2.8240776185266014) q[14];
rz(0.5361600491552279) q[14];
ry(0.08574208705797037) q[15];
rz(-0.39881781026779795) q[15];
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
ry(-1.3685798500415993) q[0];
rz(-1.6131573406130766) q[0];
ry(-1.5782548595290657) q[1];
rz(-1.4750085764820398) q[1];
ry(3.1344162663862583) q[2];
rz(-2.928856332110391) q[2];
ry(0.6106112555672041) q[3];
rz(-3.129284058392303) q[3];
ry(-0.0007971045439223445) q[4];
rz(-1.9999275505657157) q[4];
ry(0.0007211388161983518) q[5];
rz(1.0877694190939895) q[5];
ry(-0.17253351958513097) q[6];
rz(3.059526714418973) q[6];
ry(-0.19495704285832283) q[7];
rz(3.097559900241801) q[7];
ry(0.006410879620268872) q[8];
rz(1.4941482588942492) q[8];
ry(2.5591678336804082) q[9];
rz(-3.1400218698429816) q[9];
ry(-3.1263293459697263) q[10];
rz(-2.7363210649237995) q[10];
ry(3.141259871532386) q[11];
rz(2.5363462489455175) q[11];
ry(-1.4682762010284813) q[12];
rz(1.4921131680806372) q[12];
ry(-0.0022422989252781633) q[13];
rz(-2.23621762088726) q[13];
ry(-3.0040280396923484) q[14];
rz(1.4959249748562629) q[14];
ry(1.3249825428027737) q[15];
rz(-1.7217082257712066) q[15];
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
ry(3.0624349432114104) q[0];
rz(-1.3103118424791234) q[0];
ry(3.1309413838978397) q[1];
rz(1.555609747866006) q[1];
ry(0.0007669573713044203) q[2];
rz(1.931846796739261) q[2];
ry(-1.5825594077883027) q[3];
rz(-0.11551638487587423) q[3];
ry(0.2640998485532352) q[4];
rz(2.1304704306567315) q[4];
ry(-0.000729494005350162) q[5];
rz(-0.5839501151426232) q[5];
ry(1.5415893260479192) q[6];
rz(-2.932252483721482) q[6];
ry(-1.5727968274463526) q[7];
rz(-2.9276976691693144) q[7];
ry(-2.940009931935178) q[8];
rz(0.4411487100145158) q[8];
ry(1.5677047371967774) q[9];
rz(-0.014318597895196688) q[9];
ry(3.139494794003194) q[10];
rz(0.3811141508991236) q[10];
ry(-3.1379838717991335) q[11];
rz(-1.962074313648352) q[11];
ry(1.539840219644325) q[12];
rz(1.3198219540665566) q[12];
ry(3.1317776171094445) q[13];
rz(2.2673952188257758) q[13];
ry(3.1311000561168116) q[14];
rz(-2.0315651542898134) q[14];
ry(0.0017416382861821017) q[15];
rz(0.21810966207850302) q[15];
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
ry(-0.04002164544036724) q[0];
rz(3.000625469289106) q[0];
ry(2.6816689859526437) q[1];
rz(-1.7453140169161063) q[1];
ry(-1.2122432087207702) q[2];
rz(1.7756077949801075) q[2];
ry(1.0060935838286051) q[3];
rz(0.0010685128101339814) q[3];
ry(1.630496201736484) q[4];
rz(2.6451353923691876) q[4];
ry(3.043097731766741) q[5];
rz(-2.0176769348061856) q[5];
ry(2.7687058224777887) q[6];
rz(-1.8625601380343653) q[6];
ry(-2.9183139719549764) q[7];
rz(-0.42541169510511406) q[7];
ry(1.497273088247221) q[8];
rz(-0.6009771418001708) q[8];
ry(1.6024456520123698) q[9];
rz(-2.2045292442743554) q[9];
ry(1.8299657019102575) q[10];
rz(-2.141036762166192) q[10];
ry(-0.02520937805591981) q[11];
rz(-1.4311374541901385) q[11];
ry(-1.8709051431407744) q[12];
rz(0.3222828835505424) q[12];
ry(-1.5883849198455202) q[13];
rz(-2.1949198330985524) q[13];
ry(-1.4732987532262731) q[14];
rz(2.5294437734581945) q[14];
ry(-1.8844810113151862) q[15];
rz(2.7750414467236393) q[15];