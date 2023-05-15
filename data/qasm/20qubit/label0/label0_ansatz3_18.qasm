OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(3.0255404641997674) q[0];
rz(2.0671722884201005) q[0];
ry(-1.7008451080463782) q[1];
rz(0.9803709350684127) q[1];
ry(-1.0050112247493699) q[2];
rz(2.7529042623227196) q[2];
ry(0.7178007898961019) q[3];
rz(-0.5839171151938253) q[3];
ry(-2.5655128545844836) q[4];
rz(2.8414337193117793) q[4];
ry(3.099634704290527) q[5];
rz(0.5457524046139205) q[5];
ry(3.1250051704247026) q[6];
rz(-1.8723829465541453) q[6];
ry(1.010581429426308) q[7];
rz(-1.1851082115651879) q[7];
ry(-3.102343720915182) q[8];
rz(2.4334845126786755) q[8];
ry(0.16584050596429645) q[9];
rz(0.03000852645605256) q[9];
ry(0.01216283026774241) q[10];
rz(-0.972054030782145) q[10];
ry(0.7329631666908725) q[11];
rz(1.625455806276033) q[11];
ry(0.19782852302613121) q[12];
rz(0.9057542279850521) q[12];
ry(3.0687294841320307) q[13];
rz(2.976896344868146) q[13];
ry(3.064031956538681) q[14];
rz(1.6739028123513364) q[14];
ry(2.615891889695499) q[15];
rz(-0.38864003430721805) q[15];
ry(-0.008306283730471266) q[16];
rz(1.5109077565206297) q[16];
ry(3.105806498880314) q[17];
rz(2.127234364536057) q[17];
ry(1.5385719330946324) q[18];
rz(-3.1369359210206573) q[18];
ry(1.1162383042612074) q[19];
rz(0.7036964185775736) q[19];
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
ry(-0.15655284336594555) q[0];
rz(-0.48070736810499337) q[0];
ry(-0.3526939310188002) q[1];
rz(-1.0302600692936448) q[1];
ry(0.6968267248051968) q[2];
rz(-1.1751570688641744) q[2];
ry(1.5130802521216271) q[3];
rz(-3.0601144482387403) q[3];
ry(-2.6256881179136378) q[4];
rz(-2.906346820326057) q[4];
ry(3.140070711245148) q[5];
rz(-2.4368127483159965) q[5];
ry(-0.0031753036091493846) q[6];
rz(1.3325505938555098) q[6];
ry(3.0990378425445413) q[7];
rz(1.990452037363779) q[7];
ry(-0.20374693787504652) q[8];
rz(2.9788501558333382) q[8];
ry(0.0045988169626163256) q[9];
rz(2.8007530942530816) q[9];
ry(0.004744416794371506) q[10];
rz(2.8122569495226775) q[10];
ry(0.09529561803725554) q[11];
rz(-1.04989607554892) q[11];
ry(-0.8628156221236848) q[12];
rz(-0.09683978154716708) q[12];
ry(-0.03695026122503288) q[13];
rz(2.3577380568212822) q[13];
ry(-0.04251373782982615) q[14];
rz(-1.5863771370313338) q[14];
ry(2.5178857847576315) q[15];
rz(2.6870871875041322) q[15];
ry(2.739933751075105) q[16];
rz(-0.7138072369479838) q[16];
ry(3.134335792046877) q[17];
rz(2.3005766183166396) q[17];
ry(0.05447833916930023) q[18];
rz(1.3261995912413251) q[18];
ry(-0.04134367535244238) q[19];
rz(-0.547156873251355) q[19];
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
ry(3.134173446468232) q[0];
rz(-0.28113375166869137) q[0];
ry(2.098790389963335) q[1];
rz(2.8333415518885414) q[1];
ry(2.987118830674313) q[2];
rz(-1.0450361413404528) q[2];
ry(-1.559185681492733) q[3];
rz(-2.6629107466003217) q[3];
ry(-2.1266980180629) q[4];
rz(0.8077352741367262) q[4];
ry(0.06252757673667265) q[5];
rz(-0.0915438084237157) q[5];
ry(1.4758716929511548) q[6];
rz(2.396563561755691) q[6];
ry(2.1525049761804693) q[7];
rz(1.53498902595326) q[7];
ry(-2.930393230152705) q[8];
rz(2.8738255641048314) q[8];
ry(-0.030278266523884764) q[9];
rz(-0.0042582017160643915) q[9];
ry(1.947652765065448) q[10];
rz(0.19976461823109123) q[10];
ry(0.19544739870723743) q[11];
rz(-3.0976436319978036) q[11];
ry(-2.9651046206082414) q[12];
rz(-1.096929509387171) q[12];
ry(-2.8119324492009445) q[13];
rz(-3.060652812637631) q[13];
ry(2.2903807524162954) q[14];
rz(-2.0093231191858476) q[14];
ry(-1.5630472808866764) q[15];
rz(-1.8005646389221281) q[15];
ry(0.007888429395829455) q[16];
rz(-2.809272259986344) q[16];
ry(0.037050401234299646) q[17];
rz(-0.5077709692195285) q[17];
ry(-0.030310525446329284) q[18];
rz(0.7723975868031996) q[18];
ry(-1.8810947080035074) q[19];
rz(0.9014325126070686) q[19];
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
ry(0.000760638339762032) q[0];
rz(2.2989796309233324) q[0];
ry(0.583769856961256) q[1];
rz(0.71658730695985) q[1];
ry(-2.7996328429990776) q[2];
rz(3.0049435056710343) q[2];
ry(-1.1688604371401432) q[3];
rz(1.899315785611729) q[3];
ry(0.011080117522654311) q[4];
rz(2.1571678004699373) q[4];
ry(-3.099060378684673) q[5];
rz(-2.0152545473914785) q[5];
ry(-3.1375735183113527) q[6];
rz(-0.805015539942441) q[6];
ry(-2.320467612801457) q[7];
rz(0.6418028492980533) q[7];
ry(3.141012612682906) q[8];
rz(-3.0317924525457776) q[8];
ry(-0.1506721383977885) q[9];
rz(-0.12175371876643624) q[9];
ry(1.5785014921551614) q[10];
rz(0.027272191303033818) q[10];
ry(-0.2852911774210112) q[11];
rz(-0.601792660082044) q[11];
ry(3.0290316990798596) q[12];
rz(-0.1859372060213751) q[12];
ry(-3.1377037479190175) q[13];
rz(-2.4439081131592695) q[13];
ry(-3.110150846464938) q[14];
rz(-1.8870179538777974) q[14];
ry(-1.6121076384508992) q[15];
rz(2.861850152225005) q[15];
ry(3.123944058456774) q[16];
rz(-2.0470451774137377) q[16];
ry(1.2346770785506345) q[17];
rz(-1.688857038479644) q[17];
ry(-3.0211624648741857) q[18];
rz(-1.5780768004562598) q[18];
ry(-3.1242412690122) q[19];
rz(2.0858698498303543) q[19];
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
ry(-0.17747637911529157) q[0];
rz(0.1330535010475542) q[0];
ry(-1.3194215530100304) q[1];
rz(2.8030178057794815) q[1];
ry(1.5577753288935297) q[2];
rz(2.028159961839557) q[2];
ry(0.3803772353663944) q[3];
rz(2.5909087351541022) q[3];
ry(2.258283396643685) q[4];
rz(-0.9320458738147135) q[4];
ry(-3.0846843017695544) q[5];
rz(0.4514248205202958) q[5];
ry(1.4480041668204182) q[6];
rz(-0.871492786982917) q[6];
ry(3.129348113644503) q[7];
rz(-3.049718827584589) q[7];
ry(3.116823849110876) q[8];
rz(-0.42938837190948687) q[8];
ry(3.116685879426603) q[9];
rz(1.9823041861585047) q[9];
ry(1.1729389855231673) q[10];
rz(-0.050451768852722526) q[10];
ry(-1.727051276630934) q[11];
rz(1.9387617654461726) q[11];
ry(3.0190777752305697) q[12];
rz(0.6353694232645273) q[12];
ry(3.0688566173923486) q[13];
rz(-2.7788334866862243) q[13];
ry(1.5914319323043395) q[14];
rz(-2.279630835574759) q[14];
ry(-0.27508261297597514) q[15];
rz(2.9462918376760667) q[15];
ry(0.00839086969133491) q[16];
rz(-1.2824365047833046) q[16];
ry(0.04592099428279986) q[17];
rz(2.4595018503503945) q[17];
ry(-0.04690318469123662) q[18];
rz(0.5091790113959487) q[18];
ry(0.20525267884632736) q[19];
rz(0.4671535430572806) q[19];
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
ry(-3.0711628415088716) q[0];
rz(-1.9018609548528842) q[0];
ry(-1.9943217368079278) q[1];
rz(-1.8105400392317934) q[1];
ry(2.297516459457648) q[2];
rz(-2.6888842672287874) q[2];
ry(0.9808055543008674) q[3];
rz(-0.23024991641559678) q[3];
ry(3.1403062275271356) q[4];
rz(2.7135653548219363) q[4];
ry(-0.10887250083727817) q[5];
rz(-2.717441601915881) q[5];
ry(0.0015194190287359677) q[6];
rz(1.154784624665517) q[6];
ry(-3.1108276832210415) q[7];
rz(1.2033670478246512) q[7];
ry(1.8154792269726476) q[8];
rz(-3.1365852178466005) q[8];
ry(2.988101922640042) q[9];
rz(2.8184689854969402) q[9];
ry(-2.3477431156200392) q[10];
rz(1.5059248325548094) q[10];
ry(-0.4273416279714707) q[11];
rz(-2.2562939127432307) q[11];
ry(1.7895754755294884) q[12];
rz(-0.05787087424269899) q[12];
ry(-0.0011588526949735112) q[13];
rz(-2.4815727581136975) q[13];
ry(1.373610578830788) q[14];
rz(3.1064891787566067) q[14];
ry(3.046066488215585) q[15];
rz(1.1201236345716348) q[15];
ry(3.131728651157827) q[16];
rz(2.546094864050734) q[16];
ry(-0.046478759285059165) q[17];
rz(-1.816246652834641) q[17];
ry(-1.6036740393188458) q[18];
rz(2.3779784417053644) q[18];
ry(-3.1082873635846147) q[19];
rz(1.1895927158063075) q[19];
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
ry(-0.054603942356310935) q[0];
rz(1.0673120663413906) q[0];
ry(-3.0521132072473427) q[1];
rz(-2.7461173289063416) q[1];
ry(-2.067825483805251) q[2];
rz(-0.23368258230048247) q[2];
ry(1.658361439775246) q[3];
rz(-1.5023713847829014) q[3];
ry(0.7082595916363609) q[4];
rz(-2.964167124038334) q[4];
ry(-3.0640063772104305) q[5];
rz(2.994150493151951) q[5];
ry(-3.1278301399511137) q[6];
rz(-0.5423579579995166) q[6];
ry(0.029647687859188032) q[7];
rz(0.1154281387555889) q[7];
ry(-0.02763439688333769) q[8];
rz(-3.127241026692873) q[8];
ry(-0.17210460680512363) q[9];
rz(-2.6268697242708203) q[9];
ry(3.1411650087514027) q[10];
rz(-3.1215076317543846) q[10];
ry(3.0789955436376557) q[11];
rz(0.6823356260890869) q[11];
ry(3.128838469089873) q[12];
rz(-0.002035724902939087) q[12];
ry(0.1523461090663775) q[13];
rz(0.48753673990823293) q[13];
ry(-2.3129940878820605) q[14];
rz(-2.244947835245648) q[14];
ry(0.033170261198735985) q[15];
rz(3.0662439536205124) q[15];
ry(-0.00312255422604759) q[16];
rz(2.8720191970927345) q[16];
ry(-1.5210237341855317) q[17];
rz(-3.1044250728395526) q[17];
ry(-0.021278010259989603) q[18];
rz(-2.6847243376790773) q[18];
ry(-0.0024406773489348615) q[19];
rz(0.6088141432393419) q[19];
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
ry(3.103667301537416) q[0];
rz(-1.4984439590610998) q[0];
ry(1.7990624427149124) q[1];
rz(-2.757510088075006) q[1];
ry(0.937723164342671) q[2];
rz(-0.34556161692420523) q[2];
ry(-1.876213442456139) q[3];
rz(-2.8770678264908303) q[3];
ry(-0.0011286893243784846) q[4];
rz(-2.040610864977787) q[4];
ry(1.0922853973847186) q[5];
rz(-2.8849705138879354) q[5];
ry(0.002524832445295479) q[6];
rz(-3.027990347913182) q[6];
ry(3.0751727649245058) q[7];
rz(-0.6271710729283942) q[7];
ry(1.303774706060608) q[8];
rz(-1.1684475220222552) q[8];
ry(-0.17757698295500993) q[9];
rz(1.065508548057693) q[9];
ry(-1.5495546980554398) q[10];
rz(2.262820552960832) q[10];
ry(-2.5509614370800926) q[11];
rz(-0.22655870044080562) q[11];
ry(-1.898735572109459) q[12];
rz(2.326126499136093) q[12];
ry(3.1392593592636766) q[13];
rz(-2.7466430307588894) q[13];
ry(-1.7002577980434186) q[14];
rz(2.6093794694959276) q[14];
ry(2.4353343183847187e-05) q[15];
rz(1.8109167212104955) q[15];
ry(2.711566035435772) q[16];
rz(-2.440377307346607) q[16];
ry(0.18036128615996083) q[17];
rz(-0.010875720148485257) q[17];
ry(-0.057409121198631397) q[18];
rz(-1.3966986727051345) q[18];
ry(-0.0037477091955382567) q[19];
rz(-0.05242501893266205) q[19];
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
ry(0.353052757493423) q[0];
rz(1.165210026470696) q[0];
ry(-1.6340081309521315) q[1];
rz(-0.8357349546431241) q[1];
ry(-2.4857436435403106) q[2];
rz(-2.249107721888453) q[2];
ry(2.9385124517762695) q[3];
rz(-1.6342444963266611) q[3];
ry(-2.9525045968776267) q[4];
rz(0.7241395442458812) q[4];
ry(-3.0920750159336605) q[5];
rz(1.7045744857989895) q[5];
ry(-1.5605709982244118) q[6];
rz(-2.4608806441376005) q[6];
ry(-3.135845655535671) q[7];
rz(-0.12415957001286028) q[7];
ry(0.4004080895980522) q[8];
rz(-2.9430732371879964) q[8];
ry(3.1389107058862984) q[9];
rz(2.3359453894476694) q[9];
ry(-0.10219274877583118) q[10];
rz(-2.4593305205307527) q[10];
ry(-1.7845842572792066) q[11];
rz(0.40637125023433196) q[11];
ry(0.0007061550059175303) q[12];
rz(0.16446306454989834) q[12];
ry(3.0740228004390104) q[13];
rz(1.315191852224487) q[13];
ry(-3.090342157087249) q[14];
rz(1.04018799433638) q[14];
ry(-3.107894864702415) q[15];
rz(-2.9238230015923854) q[15];
ry(-0.07297511691744586) q[16];
rz(-2.2142176926170416) q[16];
ry(-1.5844056232865107) q[17];
rz(1.4464480177927945) q[17];
ry(-3.0801146356732048) q[18];
rz(0.1268552863955703) q[18];
ry(2.238618439590242) q[19];
rz(0.013329688531012351) q[19];
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
ry(3.0865908740823214) q[0];
rz(-3.0797655365014314) q[0];
ry(2.949622283167694) q[1];
rz(1.4293626090646245) q[1];
ry(3.0977914748932505) q[2];
rz(1.3984702201370682) q[2];
ry(-0.2649283101225466) q[3];
rz(-1.340104261862769) q[3];
ry(3.1391958306885144) q[4];
rz(-2.957903334031351) q[4];
ry(3.065713771176709) q[5];
rz(-0.30165659007683393) q[5];
ry(0.0018641327503488014) q[6];
rz(-3.0802788269717594) q[6];
ry(-2.9310939766587687) q[7];
rz(1.3728118570117083) q[7];
ry(0.2631788593890336) q[8];
rz(0.047206049801236015) q[8];
ry(3.02884868831593) q[9];
rz(3.025470425045605) q[9];
ry(-2.900981764587019) q[10];
rz(0.49203495603775105) q[10];
ry(2.2029776397708707) q[11];
rz(1.7898132820421209) q[11];
ry(0.49096324919873996) q[12];
rz(1.796418970905796) q[12];
ry(0.005223422786052691) q[13];
rz(0.6110797620575433) q[13];
ry(2.898357630674674) q[14];
rz(0.09798313069765931) q[14];
ry(1.9178999563588377) q[15];
rz(-1.2203293081409075) q[15];
ry(-0.008601217243286108) q[16];
rz(1.1298022267436911) q[16];
ry(-3.118282076801945) q[17];
rz(-1.963182110471911) q[17];
ry(2.9712678579221237) q[18];
rz(-2.7347741200308437) q[18];
ry(-1.6235165216279774) q[19];
rz(2.9770944586256856) q[19];
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
ry(3.112607770287129) q[0];
rz(3.034396878259477) q[0];
ry(1.6400585703273076) q[1];
rz(2.314946821829627) q[1];
ry(0.6788795494891215) q[2];
rz(0.09779292566864131) q[2];
ry(0.14688144201054107) q[3];
rz(2.5789452069624508) q[3];
ry(-2.8691965775356407) q[4];
rz(0.7724131210232446) q[4];
ry(0.010384824829221649) q[5];
rz(1.8393351277178214) q[5];
ry(3.041906585069583) q[6];
rz(-2.2671548816423996) q[6];
ry(0.0008173312086758261) q[7];
rz(-0.34783889580979727) q[7];
ry(1.42040350094344) q[8];
rz(-0.8023138202926133) q[8];
ry(-3.114367664143653) q[9];
rz(0.11591104803748653) q[9];
ry(-0.018144332090940907) q[10];
rz(2.554081030342337) q[10];
ry(2.814399143199645) q[11];
rz(-1.2439416742621832) q[11];
ry(3.1389316156167126) q[12];
rz(-1.6429939351194398) q[12];
ry(3.069151339581751) q[13];
rz(2.7189224987708056) q[13];
ry(0.26597688073583114) q[14];
rz(-3.0635851125511038) q[14];
ry(3.0743825959761057) q[15];
rz(1.7998665863184922) q[15];
ry(-3.0299668740244674) q[16];
rz(2.696219004708418) q[16];
ry(-0.015254033217476332) q[17];
rz(0.5468431793416774) q[17];
ry(1.574649422303241) q[18];
rz(-2.274569788730926) q[18];
ry(-0.910462750743994) q[19];
rz(2.7629862609911315) q[19];
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
ry(-0.06676279456795786) q[0];
rz(-1.511771841523351) q[0];
ry(0.21718989191331506) q[1];
rz(-0.4584980877806125) q[1];
ry(-3.0076831265224193) q[2];
rz(0.694894528794305) q[2];
ry(0.627374719192095) q[3];
rz(-3.1034261731756447) q[3];
ry(-3.1402283042444163) q[4];
rz(0.03502683097064804) q[4];
ry(-2.0066171805251996) q[5];
rz(2.9881731060992256) q[5];
ry(-3.1401385370282426) q[6];
rz(0.16592223475582918) q[6];
ry(2.329414464657229) q[7];
rz(0.5829491427553128) q[7];
ry(3.129870045048049) q[8];
rz(-0.91603324372301) q[8];
ry(-1.6272681562942506) q[9];
rz(3.0831234235047362) q[9];
ry(-2.477276985530834) q[10];
rz(-1.3993060424108141) q[10];
ry(2.67839183082935) q[11];
rz(-2.4561375305673363) q[11];
ry(-0.40960072703453765) q[12];
rz(-1.7252121986377897) q[12];
ry(-0.008872088675321974) q[13];
rz(1.315121226751543) q[13];
ry(-0.5866477382250936) q[14];
rz(2.4524086012847404) q[14];
ry(-0.24649387938512388) q[15];
rz(-2.0866471970398446) q[15];
ry(3.070205250212596) q[16];
rz(-1.7490011080653236) q[16];
ry(1.487834485424971) q[17];
rz(-2.4177201919827755) q[17];
ry(-1.5949662672294522) q[18];
rz(2.9969481631450408) q[18];
ry(1.005736725322362) q[19];
rz(-2.311107554482693) q[19];
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
ry(0.23980818763112666) q[0];
rz(-2.120634305173209) q[0];
ry(1.644687257556667) q[1];
rz(2.93456838717432) q[1];
ry(1.3378203986474633) q[2];
rz(-0.4366771717599702) q[2];
ry(2.900845684555271) q[3];
rz(0.6996536143991962) q[3];
ry(-1.4716721105013706) q[4];
rz(2.4119105210752667) q[4];
ry(0.07272906784172938) q[5];
rz(-3.091347246250593) q[5];
ry(-1.5381506323300034) q[6];
rz(-2.548725802761843) q[6];
ry(-3.1413038937816213) q[7];
rz(2.369947462773405) q[7];
ry(1.389484403281495) q[8];
rz(1.586661999824911) q[8];
ry(0.10733575427585886) q[9];
rz(0.1343234990883886) q[9];
ry(-3.1191862682475233) q[10];
rz(0.15617483921503367) q[10];
ry(-3.1383801790860106) q[11];
rz(0.9919431360843795) q[11];
ry(0.6106900166376188) q[12];
rz(2.2067032590291293) q[12];
ry(-2.3280283934318757) q[13];
rz(-0.7012280665389153) q[13];
ry(-2.873076016279513) q[14];
rz(-1.3327209837874205) q[14];
ry(0.03055691283974763) q[15];
rz(2.365414570733883) q[15];
ry(3.1386530003618205) q[16];
rz(1.2472316682613451) q[16];
ry(0.03119253601276051) q[17];
rz(-2.0408768269344604) q[17];
ry(3.1369751607958887) q[18];
rz(2.9743788486766785) q[18];
ry(2.018532284422684) q[19];
rz(-3.1326140848677206) q[19];
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
ry(3.1268429412549716) q[0];
rz(0.6106793487299935) q[0];
ry(2.625895450289436) q[1];
rz(3.0705153551164055) q[1];
ry(-1.188065783097943) q[2];
rz(1.60908721464628) q[2];
ry(-1.4923122034385203) q[3];
rz(1.0369238598196482) q[3];
ry(0.00283238364044891) q[4];
rz(2.1318996310875136) q[4];
ry(3.006363171897643) q[5];
rz(-2.4115855546165785) q[5];
ry(-4.182946960567201e-05) q[6];
rz(2.8427304591805815) q[6];
ry(-1.601385138319026) q[7];
rz(0.17799608421075233) q[7];
ry(2.944968292615658) q[8];
rz(0.08475993357249043) q[8];
ry(2.1641317667047777) q[9];
rz(0.1935343877005769) q[9];
ry(0.36285337357374503) q[10];
rz(0.19641092981343933) q[10];
ry(3.1373877848274088) q[11];
rz(2.315927386886303) q[11];
ry(0.0021043884753712445) q[12];
rz(1.4929262273312425) q[12];
ry(-3.1411617060508514) q[13];
rz(-2.355366180477097) q[13];
ry(-0.0010747421315695505) q[14];
rz(2.579891963624152) q[14];
ry(-0.24097439137771737) q[15];
rz(-3.0770897687119083) q[15];
ry(3.076017927640258) q[16];
rz(-0.6308464139834968) q[16];
ry(1.1644181516453003) q[17];
rz(2.427139461340729) q[17];
ry(-1.4483646363579536) q[18];
rz(-0.7796491278625206) q[18];
ry(-0.7169180760756304) q[19];
rz(2.9902232907772692) q[19];
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
ry(0.04238657866572826) q[0];
rz(1.7310404061842641) q[0];
ry(0.3251848248198887) q[1];
rz(1.3272476056463534) q[1];
ry(2.9990386162610627) q[2];
rz(-0.2098228013459078) q[2];
ry(-0.018451164708903987) q[3];
rz(-1.0030683851284128) q[3];
ry(-1.4091449495167876) q[4];
rz(-2.1542960167522462) q[4];
ry(-0.2520937786040358) q[5];
rz(-3.1397618581807514) q[5];
ry(1.952289751939425) q[6];
rz(1.233456001738868) q[6];
ry(-3.1410015526670643) q[7];
rz(-1.3998293314851367) q[7];
ry(-0.2145897963698975) q[8];
rz(-1.5444029850350631) q[8];
ry(0.007267459227248826) q[9];
rz(0.45776441751000213) q[9];
ry(2.343395507025816) q[10];
rz(-2.8507353724774247) q[10];
ry(-0.0006857920325726852) q[11];
rz(3.1010838292556233) q[11];
ry(0.938557866563441) q[12];
rz(-2.9655726818298365) q[12];
ry(0.6435559750011929) q[13];
rz(2.4943562329144973) q[13];
ry(-1.7428516154918974) q[14];
rz(1.0396684322890606) q[14];
ry(-0.9566121371132138) q[15];
rz(3.005610507431435) q[15];
ry(-2.9678676075433916) q[16];
rz(2.4398052505481225) q[16];
ry(1.5329643583031265) q[17];
rz(0.7058095714588939) q[17];
ry(3.072864195270029) q[18];
rz(2.304178172470834) q[18];
ry(1.8967372886813036) q[19];
rz(-1.0720780268750463) q[19];
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
ry(0.1580504580803803) q[0];
rz(0.26420986029519955) q[0];
ry(2.360958038176831) q[1];
rz(-2.431612574664763) q[1];
ry(-1.572494467283855) q[2];
rz(-0.06447582652712903) q[2];
ry(-2.2907001105494205) q[3];
rz(0.01177355437773997) q[3];
ry(-3.1378715199893747) q[4];
rz(-0.6875824965604813) q[4];
ry(2.9337315386014056) q[5];
rz(1.9171846341304468) q[5];
ry(5.0907194934592616e-05) q[6];
rz(-2.6796816889511255) q[6];
ry(1.5661720413023907) q[7];
rz(0.2283568589678131) q[7];
ry(-3.1387495150895686) q[8];
rz(-3.0147523828412677) q[8];
ry(0.22838159998564622) q[9];
rz(0.3736181220919325) q[9];
ry(-2.9412831201682677) q[10];
rz(-2.8145079630129377) q[10];
ry(-0.01015743897321275) q[11];
rz(2.705211703710672) q[11];
ry(-0.00375062398657336) q[12];
rz(-0.46064761097544177) q[12];
ry(0.0015696896131107481) q[13];
rz(0.10555515225395457) q[13];
ry(3.1402154161743354) q[14];
rz(-3.1108254015185928) q[14];
ry(-0.008056620595375001) q[15];
rz(0.044320214746901065) q[15];
ry(-0.06524906006815988) q[16];
rz(-2.2793519181327646) q[16];
ry(3.133227274806609) q[17];
rz(-2.4396457755726186) q[17];
ry(-2.7065484835371825) q[18];
rz(0.2098637664880778) q[18];
ry(0.005243880331778249) q[19];
rz(-2.0641718973334875) q[19];
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
ry(0.814680192524766) q[0];
rz(-2.300417521110832) q[0];
ry(1.1544900140016816) q[1];
rz(-3.1393201852352255) q[1];
ry(-0.016852106883443928) q[2];
rz(0.35165444590121686) q[2];
ry(-0.2733084465644233) q[3];
rz(-2.2240448769105248) q[3];
ry(3.0509312660448082) q[4];
rz(2.7871300859017882) q[4];
ry(3.1385379691577624) q[5];
rz(2.8690798132473323) q[5];
ry(1.5942584621297355) q[6];
rz(2.438448053733158) q[6];
ry(3.1412412497687225) q[7];
rz(-1.6176190116370497) q[7];
ry(-1.4982457238286677) q[8];
rz(-0.18762710419982942) q[8];
ry(3.139426201709428) q[9];
rz(0.9318183249789874) q[9];
ry(-0.7721097970881274) q[10];
rz(-1.796199247723942) q[10];
ry(-3.104586315319437) q[11];
rz(2.643268290305631) q[11];
ry(-1.6851338121700905) q[12];
rz(1.5950948882014266) q[12];
ry(-2.2048539005393666) q[13];
rz(2.8421249004100693) q[13];
ry(1.5199077308407722) q[14];
rz(2.314468055511367) q[14];
ry(-2.4711021535840154) q[15];
rz(0.8461359184296021) q[15];
ry(1.9298940928838046) q[16];
rz(0.7167619804367966) q[16];
ry(1.5750507293303846) q[17];
rz(2.2608345413133977) q[17];
ry(-1.3761946472037065) q[18];
rz(3.014893764263495) q[18];
ry(1.9330213920332648) q[19];
rz(1.9355959943703045) q[19];
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
ry(0.4065544494659324) q[0];
rz(0.6919201750473344) q[0];
ry(1.0895256068218817) q[1];
rz(3.0017439669716244) q[1];
ry(0.12453601619994181) q[2];
rz(2.5124946832653228) q[2];
ry(1.3208097560054795) q[3];
rz(-0.6318042261353095) q[3];
ry(1.1069658973782366) q[4];
rz(-0.10305566422102838) q[4];
ry(0.0654295389636399) q[5];
rz(-1.6872045963288889) q[5];
ry(3.138440913810724) q[6];
rz(-2.778069445716138) q[6];
ry(-1.6734163396269404) q[7];
rz(-1.2937745851048161) q[7];
ry(-2.7588052336480433) q[8];
rz(0.013913575254112385) q[8];
ry(-1.5870919038903977) q[9];
rz(-0.03882441658749568) q[9];
ry(-1.2656574088412729) q[10];
rz(1.1817532303480407) q[10];
ry(3.0098799896621378) q[11];
rz(2.9564997600677407) q[11];
ry(0.2839479949552058) q[12];
rz(-1.4920429505250157) q[12];
ry(3.130696197024194) q[13];
rz(2.8963750240412427) q[13];
ry(-3.140496886495675) q[14];
rz(-2.2025708864736866) q[14];
ry(-3.0905420522333147) q[15];
rz(-0.4925387775626522) q[15];
ry(-3.1388488838559176) q[16];
rz(2.104541622810535) q[16];
ry(-1.471918030273837) q[17];
rz(3.098050537124948) q[17];
ry(0.02506169484141907) q[18];
rz(-2.9281957111358463) q[18];
ry(1.5510814829941835) q[19];
rz(1.1765852740860356) q[19];
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
ry(-1.36915883803386) q[0];
rz(-2.1053832456483406) q[0];
ry(-2.085894688374565) q[1];
rz(-0.08003832302198392) q[1];
ry(3.1122172297436346) q[2];
rz(-0.4946861785537212) q[2];
ry(3.119790979376702) q[3];
rz(-1.8801449934342382) q[3];
ry(1.7553684268545826) q[4];
rz(-0.9305734136790411) q[4];
ry(0.0007205790820350444) q[5];
rz(-2.369207375080339) q[5];
ry(3.11241231176634) q[6];
rz(-0.6551811652692887) q[6];
ry(-1.3580108625465899) q[7];
rz(3.078723189655416) q[7];
ry(0.6974249936849173) q[8];
rz(-1.1633685226952104) q[8];
ry(-0.11725171934245072) q[9];
rz(-0.19861709231065125) q[9];
ry(0.012019319879837731) q[10];
rz(2.045814344219265) q[10];
ry(2.9395386360701123) q[11];
rz(-2.7213363047416532) q[11];
ry(3.0556125120097173) q[12];
rz(-1.5543438069067517) q[12];
ry(-0.07808315158095422) q[13];
rz(-1.1460708011682206) q[13];
ry(-2.7049348885570343) q[14];
rz(2.642024776847) q[14];
ry(-1.554804567084967) q[15];
rz(-1.4658369478183753) q[15];
ry(1.4367828895384926) q[16];
rz(2.4111675457931994) q[16];
ry(-2.9405758761964234) q[17];
rz(0.08925018275475873) q[17];
ry(-1.5333203870068053) q[18];
rz(-1.329138191394943) q[18];
ry(-3.1389187886356615) q[19];
rz(-1.8452181216618249) q[19];
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
ry(-1.5213538416775945) q[0];
rz(3.098230143373667) q[0];
ry(-1.8634789608329712) q[1];
rz(3.101145609130315) q[1];
ry(2.763255590717164) q[2];
rz(-0.6003680568132825) q[2];
ry(-2.129762770506735) q[3];
rz(-2.8528678861136196) q[3];
ry(3.1115114638536596) q[4];
rz(0.011369886893634273) q[4];
ry(-3.128056453497995) q[5];
rz(1.4485274400363652) q[5];
ry(2.916801684438862) q[6];
rz(-2.904516480102693) q[6];
ry(0.18925831627457818) q[7];
rz(-3.0800073831454493) q[7];
ry(-1.5130734919817597) q[8];
rz(0.7654339540297576) q[8];
ry(-3.111857476373983) q[9];
rz(0.8532293232476816) q[9];
ry(1.9984560152596453) q[10];
rz(-0.874036108028023) q[10];
ry(-3.0964703804224762) q[11];
rz(1.966256549743876) q[11];
ry(-2.870121434106491) q[12];
rz(2.8599887893114557) q[12];
ry(3.1337444059462776) q[13];
rz(0.17203617644540106) q[13];
ry(-0.0016002733357902699) q[14];
rz(1.0498285748196823) q[14];
ry(-3.0498933472198853) q[15];
rz(-0.28522545846519165) q[15];
ry(0.08290257334368523) q[16];
rz(-0.8569999683100075) q[16];
ry(-1.4834944887995176) q[17];
rz(-0.30673892449425844) q[17];
ry(0.30971214705733235) q[18];
rz(-2.487081219019528) q[18];
ry(-1.7715752344785654) q[19];
rz(-1.5699963611708185) q[19];
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
ry(2.2393231181420346) q[0];
rz(1.6608796998194943) q[0];
ry(-1.5749169431151793) q[1];
rz(3.0995918759081773) q[1];
ry(-0.020133219868583296) q[2];
rz(-2.3759914555399844) q[2];
ry(-0.002584381708903294) q[3];
rz(-2.1370550368233796) q[3];
ry(-3.1187509555377764) q[4];
rz(-0.7640767917477644) q[4];
ry(3.141010013078437) q[5];
rz(-0.24798133754410315) q[5];
ry(-3.1312455131595684) q[6];
rz(1.8568601524211115) q[6];
ry(1.7760416782909871) q[7];
rz(-1.5937897275666153) q[7];
ry(-0.017672262113935133) q[8];
rz(0.9960698685138261) q[8];
ry(-2.9934206600393543) q[9];
rz(-0.6499728326367776) q[9];
ry(-0.03246015984618378) q[10];
rz(0.16712192865914677) q[10];
ry(2.930474504232704) q[11];
rz(-3.12547066915153) q[11];
ry(-0.09378919527511352) q[12];
rz(-2.6986980567122623) q[12];
ry(2.8355632244097566) q[13];
rz(3.016909670973652) q[13];
ry(2.968293125576272) q[14];
rz(-1.1178937732348944) q[14];
ry(-1.8453567782514844) q[15];
rz(2.7913984591746543) q[15];
ry(-3.1301445685482747) q[16];
rz(1.2241352835309045) q[16];
ry(-0.04516368245886945) q[17];
rz(-2.8653532286748304) q[17];
ry(-0.05469667585323368) q[18];
rz(0.9026922619825681) q[18];
ry(1.677549416331826) q[19];
rz(-0.47808399150523256) q[19];
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
ry(-0.07966313020871593) q[0];
rz(-0.055857379354296105) q[0];
ry(-1.6381235767529152) q[1];
rz(1.611669637957995) q[1];
ry(-1.725503877002951) q[2];
rz(0.004513785110688673) q[2];
ry(0.5388648347532434) q[3];
rz(-2.4786938192778787) q[3];
ry(-1.3203678965482224) q[4];
rz(2.0811919662510983) q[4];
ry(-1.5444389135243466) q[5];
rz(-3.139898905705484) q[5];
ry(-1.566091406432453) q[6];
rz(0.04979005484152932) q[6];
ry(-1.5622242995086872) q[7];
rz(0.08281647323017591) q[7];
ry(-2.5212964011782217) q[8];
rz(1.3992333975009146) q[8];
ry(-1.572104138895634) q[9];
rz(-0.019678923254185573) q[9];
ry(-0.22801599718310306) q[10];
rz(-0.8738081373197273) q[10];
ry(1.5696972718517068) q[11];
rz(-3.136397902612321) q[11];
ry(1.6582596516612627) q[12];
rz(3.111785752192697) q[12];
ry(-1.3067862597312525) q[13];
rz(-3.0399505650031746) q[13];
ry(1.86233205415145) q[14];
rz(0.06668234761972337) q[14];
ry(1.3993910385664496) q[15];
rz(3.042504108352637) q[15];
ry(-1.6044198764527369) q[16];
rz(-0.07524459377439147) q[16];
ry(0.15371274872311247) q[17];
rz(-1.649147303370466) q[17];
ry(-1.56983905817447) q[18];
rz(-0.01851335344076831) q[18];
ry(-0.03685822274894246) q[19];
rz(2.0530127088423926) q[19];