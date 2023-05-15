OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.146880642551797) q[0];
rz(2.6183198373484147) q[0];
ry(2.9073929631239324) q[1];
rz(2.4760906329480896) q[1];
ry(2.867540670736456) q[2];
rz(-2.5059102480262796) q[2];
ry(-0.024189280435894456) q[3];
rz(-2.467729856895917) q[3];
ry(-1.6263466020431563) q[4];
rz(1.3827826950815507) q[4];
ry(2.255698243337414) q[5];
rz(1.6503147572027699) q[5];
ry(-0.37377563862785823) q[6];
rz(-1.8081892940447002) q[6];
ry(-3.13290885853685) q[7];
rz(1.5365228304301342) q[7];
ry(2.5947235698354145) q[8];
rz(-0.6272735337342361) q[8];
ry(0.09847754531064101) q[9];
rz(1.6062458016343495) q[9];
ry(-2.5835336842294283) q[10];
rz(-2.340923770622418) q[10];
ry(2.3039401494902227) q[11];
rz(0.4381263255593896) q[11];
ry(2.2100146917321712) q[12];
rz(-0.9337771706111359) q[12];
ry(2.9583285778506814) q[13];
rz(-1.2013966778875371) q[13];
ry(2.142066859114882) q[14];
rz(-1.016858557536382) q[14];
ry(3.095121426623958) q[15];
rz(0.05935949020602376) q[15];
ry(-0.025127999668325473) q[16];
rz(0.2617164140865728) q[16];
ry(-0.09362784269514368) q[17];
rz(1.2451622191999343) q[17];
ry(-1.5605034915987614) q[18];
rz(-2.2636014861979494) q[18];
ry(1.1815302784587356) q[19];
rz(-0.18832024652408882) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.1185959323824894) q[0];
rz(-1.0050612265975616) q[0];
ry(-1.3876822294649946) q[1];
rz(-0.8297312564721063) q[1];
ry(-2.553252090943401) q[2];
rz(-1.331940669467746) q[2];
ry(2.8960950680269852) q[3];
rz(-0.14192006782188948) q[3];
ry(0.8687816452810422) q[4];
rz(0.9052445217778784) q[4];
ry(-3.138114858367797) q[5];
rz(2.6644175685445224) q[5];
ry(1.485472406955565) q[6];
rz(0.06855542861085341) q[6];
ry(-3.1234521718690047) q[7];
rz(-2.1344485805414113) q[7];
ry(-0.4715730820452162) q[8];
rz(-1.3720850132166706) q[8];
ry(1.778631998551213) q[9];
rz(-0.05713756990775031) q[9];
ry(1.3386323653002536) q[10];
rz(2.9572227720988353) q[10];
ry(-0.3381532460229942) q[11];
rz(2.533475217804383) q[11];
ry(-0.44333883590706497) q[12];
rz(0.22673667448896798) q[12];
ry(2.6790346978774107) q[13];
rz(2.864605480049294) q[13];
ry(-2.2256145839642056) q[14];
rz(-1.6305312003081671) q[14];
ry(2.784893883303358) q[15];
rz(-1.7133945961929018) q[15];
ry(-3.124350717528424) q[16];
rz(0.5824519251022836) q[16];
ry(-2.914347379436412) q[17];
rz(-2.77763959720042) q[17];
ry(2.456799387205361) q[18];
rz(-2.2799986952529485) q[18];
ry(-2.9963453194533005) q[19];
rz(1.961863351239123) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.4157985519138077) q[0];
rz(-2.9756700922874826) q[0];
ry(0.18229326026188525) q[1];
rz(1.0603134790896545) q[1];
ry(0.00017137341055069896) q[2];
rz(0.8240960616273092) q[2];
ry(3.1023169620794815) q[3];
rz(-0.9169028265467913) q[3];
ry(2.754654750052368) q[4];
rz(-1.817278795965633) q[4];
ry(-3.0401861279044655) q[5];
rz(-1.9957805914281135) q[5];
ry(2.5642847122635053) q[6];
rz(3.1160618888240634) q[6];
ry(-0.001417139985842891) q[7];
rz(-1.1395156627322958) q[7];
ry(-2.414864425722704) q[8];
rz(-2.967931143896224) q[8];
ry(2.160468597038857) q[9];
rz(-1.7216841596143375) q[9];
ry(-2.9408734408887223) q[10];
rz(-2.5736959929531054) q[10];
ry(1.855330276078302) q[11];
rz(-2.1530794830702433) q[11];
ry(0.40929378441114406) q[12];
rz(-2.660530363030777) q[12];
ry(-2.9051003157092596) q[13];
rz(2.9181882993098367) q[13];
ry(2.0303822320690283) q[14];
rz(-2.427576130478725) q[14];
ry(-0.047482956352162375) q[15];
rz(-1.5659047118860059) q[15];
ry(-3.1153420115983823) q[16];
rz(-2.2763965587976145) q[16];
ry(-3.101376158057868) q[17];
rz(-1.7852722676813848) q[17];
ry(1.2119428601057263) q[18];
rz(1.9486014821615305) q[18];
ry(2.3050831627474526) q[19];
rz(2.50609829027986) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.847189002111086) q[0];
rz(-0.6000150164560479) q[0];
ry(2.213785125490902) q[1];
rz(2.4114469052994303) q[1];
ry(1.1698548018382984) q[2];
rz(0.935240281274786) q[2];
ry(2.861595746622266) q[3];
rz(-0.3277386594412176) q[3];
ry(-2.9328688955583497) q[4];
rz(2.1723840555752285) q[4];
ry(3.133126272350299) q[5];
rz(-0.6593014729096209) q[5];
ry(1.6797352680552244) q[6];
rz(-0.5824801240433799) q[6];
ry(-1.472207704682738) q[7];
rz(1.5295593229143325) q[7];
ry(0.6935909912166673) q[8];
rz(0.4004516903271815) q[8];
ry(-0.45243592540882493) q[9];
rz(-1.693258022269625) q[9];
ry(0.04886383085915442) q[10];
rz(-0.6490660783448677) q[10];
ry(-0.07316025496479013) q[11];
rz(2.935780161342855) q[11];
ry(-1.4474419077738505) q[12];
rz(0.5987736149179866) q[12];
ry(0.9358345302861821) q[13];
rz(-0.1499181884710421) q[13];
ry(2.804418910687426) q[14];
rz(-1.2040885084316049) q[14];
ry(-0.3389162879159633) q[15];
rz(-2.874779103639757) q[15];
ry(0.018484859694822834) q[16];
rz(1.0509346556391028) q[16];
ry(2.4846869028166294) q[17];
rz(-1.0758235252296382) q[17];
ry(-2.782897881017645) q[18];
rz(-2.0060309965872927) q[18];
ry(1.3304618671942752) q[19];
rz(-2.5543391197160457) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.03015797967449) q[0];
rz(1.0875200644764405) q[0];
ry(2.1288174872278764) q[1];
rz(0.35254901712044345) q[1];
ry(0.24279607596818256) q[2];
rz(1.997052785921984) q[2];
ry(-3.1376690795536772) q[3];
rz(2.2233555602923305) q[3];
ry(1.7113583827567491) q[4];
rz(-0.8492675005997325) q[4];
ry(2.0490581395004694) q[5];
rz(-2.216198990522888) q[5];
ry(0.7495980783549792) q[6];
rz(-2.9276153454816733) q[6];
ry(1.752085053967538) q[7];
rz(-3.1211960742880174) q[7];
ry(1.889654437312394) q[8];
rz(0.05452466533473937) q[8];
ry(-0.42177648034406356) q[9];
rz(-0.6810655126870123) q[9];
ry(-2.9238787962015462) q[10];
rz(-1.0583783000441107) q[10];
ry(2.4238313385609187) q[11];
rz(-2.7068208453655194) q[11];
ry(0.005605303790514604) q[12];
rz(2.4759425249929814) q[12];
ry(-2.547896547432895) q[13];
rz(0.6945297499768258) q[13];
ry(1.097433176284734) q[14];
rz(2.1769801055979423) q[14];
ry(-2.462680145309873) q[15];
rz(-0.10073752050222318) q[15];
ry(0.8734060747030393) q[16];
rz(2.014628261318972) q[16];
ry(0.8595976646235711) q[17];
rz(-0.5888136873154494) q[17];
ry(0.1466180970803599) q[18];
rz(2.303185168954354) q[18];
ry(-0.3658205252658597) q[19];
rz(1.4717397718923806) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.3051041246142274) q[0];
rz(2.0729634161038764) q[0];
ry(-3.084310187387811) q[1];
rz(3.0612312877135186) q[1];
ry(2.9969635207321486) q[2];
rz(1.729986230597687) q[2];
ry(0.11060800794082423) q[3];
rz(-2.9047002186259188) q[3];
ry(-2.777038604987216) q[4];
rz(0.9073236585678425) q[4];
ry(-0.03840904549181535) q[5];
rz(2.4872668256172807) q[5];
ry(1.4335069976553803) q[6];
rz(3.116928214969218) q[6];
ry(0.6619763338124836) q[7];
rz(0.013716862587450864) q[7];
ry(2.8961007720034195) q[8];
rz(3.0662084158533363) q[8];
ry(0.55425053512961) q[9];
rz(-0.27490356243742475) q[9];
ry(0.11711388619945898) q[10];
rz(-2.1095272482537144) q[10];
ry(-1.5494161439684742) q[11];
rz(0.3687931044354273) q[11];
ry(-0.07720607806995501) q[12];
rz(0.965393683773688) q[12];
ry(-1.0231926888420624) q[13];
rz(1.1585197129173022) q[13];
ry(-2.7209691429411276) q[14];
rz(-0.32843499400382536) q[14];
ry(0.02158100405078133) q[15];
rz(1.7110819776726705) q[15];
ry(-3.1191083757523237) q[16];
rz(-2.0559657544846517) q[16];
ry(1.7304458377864618) q[17];
rz(-1.441338245791576) q[17];
ry(2.8069791989763644) q[18];
rz(1.1435118238590973) q[18];
ry(-2.3857642756393753) q[19];
rz(-2.169769799082254) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.7588748776545984) q[0];
rz(1.04912404978979) q[0];
ry(2.362911857953847) q[1];
rz(-3.0144466163807295) q[1];
ry(-2.7896801830153377) q[2];
rz(-2.729046413491233) q[2];
ry(3.1366032918435924) q[3];
rz(-0.34741209875556467) q[3];
ry(2.8860867148348763) q[4];
rz(0.12747940802930646) q[4];
ry(0.611727740360454) q[5];
rz(-1.21856080093007) q[5];
ry(-0.8531250854356939) q[6];
rz(0.1535686328267598) q[6];
ry(0.6199237855694788) q[7];
rz(3.1208453605837234) q[7];
ry(0.49348176001152044) q[8];
rz(2.3702831806715463) q[8];
ry(-2.359423728914393) q[9];
rz(-0.1268150114764701) q[9];
ry(3.1313533026023337) q[10];
rz(0.22152990138997186) q[10];
ry(1.2862327439150734) q[11];
rz(1.127391434654502) q[11];
ry(2.842479133056893) q[12];
rz(-1.5282753342888238) q[12];
ry(1.5548150640791165) q[13];
rz(-2.174901345076854) q[13];
ry(1.7610034926659839) q[14];
rz(-1.3313149411614686) q[14];
ry(-1.5178745989786) q[15];
rz(-1.0206185529457208) q[15];
ry(-2.514218312662712) q[16];
rz(-0.6319433741588136) q[16];
ry(-0.03533079853792664) q[17];
rz(0.8588547335864076) q[17];
ry(0.0015042319426434858) q[18];
rz(-0.82506473775309) q[18];
ry(-0.48581225408983736) q[19];
rz(2.7331725469446924) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.960418377725607) q[0];
rz(0.44881868835171657) q[0];
ry(0.006227767617322288) q[1];
rz(-1.8338984146427757) q[1];
ry(-2.858412481279987) q[2];
rz(2.9778045425315227) q[2];
ry(3.126160118669705) q[3];
rz(0.9639654393069836) q[3];
ry(1.026681873251306) q[4];
rz(-1.3040630492286107) q[4];
ry(-2.748243262424414) q[5];
rz(-0.8615880667928906) q[5];
ry(0.3378304485982246) q[6];
rz(-0.16786627588968717) q[6];
ry(-1.7728914193587304) q[7];
rz(0.7946222699299001) q[7];
ry(-3.1404655097529304) q[8];
rz(2.7013089893233304) q[8];
ry(-1.0987541124975335) q[9];
rz(0.08462539075530184) q[9];
ry(-0.014312191170979597) q[10];
rz(-1.4478582377873748) q[10];
ry(1.1462570297762642) q[11];
rz(-2.1876626119925167) q[11];
ry(2.9938143807072257) q[12];
rz(-1.3265096163663301) q[12];
ry(2.2630296493560422) q[13];
rz(2.214989680933658) q[13];
ry(-0.5248648884519498) q[14];
rz(-0.6250013068267473) q[14];
ry(-0.10857125163131087) q[15];
rz(-2.7246770609751865) q[15];
ry(1.4669542372008522) q[16];
rz(-0.18573479688406414) q[16];
ry(-1.0809884918072585) q[17];
rz(-1.6404845880448065) q[17];
ry(-2.502191083454612) q[18];
rz(2.681708054136861) q[18];
ry(0.11803192996095993) q[19];
rz(2.93589470689787) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.6886896203182413) q[0];
rz(-3.0356772655766857) q[0];
ry(1.297091702745047) q[1];
rz(-2.7083316565937317) q[1];
ry(0.4046689378756243) q[2];
rz(-0.8324527417389226) q[2];
ry(-0.010171721854354865) q[3];
rz(1.479776418737301) q[3];
ry(0.3146096053697295) q[4];
rz(-2.875592424832708) q[4];
ry(-1.9425361307965658) q[5];
rz(-0.846032010152206) q[5];
ry(-3.043692702490378) q[6];
rz(-3.0030698344736364) q[6];
ry(-2.991436698266134) q[7];
rz(0.9677283701702217) q[7];
ry(-0.09671167655543211) q[8];
rz(0.06507401166079863) q[8];
ry(2.999032486623051) q[9];
rz(-2.5210530477205326) q[9];
ry(-1.5429922422840883) q[10];
rz(-1.9138169416224111) q[10];
ry(-0.5241127574729152) q[11];
rz(1.2303854466952813) q[11];
ry(-0.14341428091118222) q[12];
rz(2.9988374190855174) q[12];
ry(-0.7837367454211315) q[13];
rz(-0.8811980677394247) q[13];
ry(2.582044471719133) q[14];
rz(-0.9763326750334969) q[14];
ry(-3.114941014411099) q[15];
rz(-1.2494505708952188) q[15];
ry(2.242486723902873) q[16];
rz(1.3993313092640807) q[16];
ry(0.30969585363928864) q[17];
rz(-0.8088392994169817) q[17];
ry(0.9735704233588862) q[18];
rz(1.7868453812522223) q[18];
ry(-2.275040717440552) q[19];
rz(-3.0414248081332755) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.3523630394984029) q[0];
rz(0.37745104111063116) q[0];
ry(-2.42892523351237) q[1];
rz(2.898574849513214) q[1];
ry(-1.6184468802635483) q[2];
rz(1.3489580065115607) q[2];
ry(-0.03189752342188523) q[3];
rz(-2.691554123161399) q[3];
ry(-2.754137048690782) q[4];
rz(2.9259270686539525) q[4];
ry(-0.9564866400661911) q[5];
rz(-0.8666913236192626) q[5];
ry(3.1163142574923746) q[6];
rz(-1.9572874615868694) q[6];
ry(-2.138688542752595) q[7];
rz(-3.0701558063538963) q[7];
ry(3.113813583462562) q[8];
rz(0.3563845957892013) q[8];
ry(-0.03586016305882023) q[9];
rz(-2.3148210106948883) q[9];
ry(0.38980710340320623) q[10];
rz(-2.465682173156431) q[10];
ry(-1.5650233029263274) q[11];
rz(-0.7972496215647945) q[11];
ry(0.10235365490483066) q[12];
rz(-2.414655105360788) q[12];
ry(-0.27358186901441695) q[13];
rz(0.0431913943382924) q[13];
ry(-0.354050430849566) q[14];
rz(-2.0255499432822224) q[14];
ry(-0.5057158034737353) q[15];
rz(0.2265238294276533) q[15];
ry(2.345940364452071) q[16];
rz(1.4374560737523772) q[16];
ry(0.0100428027991315) q[17];
rz(-1.4016503351845513) q[17];
ry(3.0710476512792924) q[18];
rz(-2.6844540279216607) q[18];
ry(1.3475655256405492) q[19];
rz(1.7465581002400654) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.2413392406784265) q[0];
rz(-1.2342723697810742) q[0];
ry(-1.2799584221294695) q[1];
rz(3.129321175076559) q[1];
ry(0.024726445870001782) q[2];
rz(-0.9482587240235762) q[2];
ry(3.0808974644529394) q[3];
rz(-2.909497536226923) q[3];
ry(1.1181104105875919) q[4];
rz(-1.9855639425676421) q[4];
ry(-1.3661663471696472) q[5];
rz(-0.4355528205136725) q[5];
ry(0.009963097062152038) q[6];
rz(1.8177198014935005) q[6];
ry(1.2315733420087804) q[7];
rz(-3.0567725580092207) q[7];
ry(2.495298979977459) q[8];
rz(-1.528880478392006) q[8];
ry(-2.9687872818754037) q[9];
rz(1.9608538732293581) q[9];
ry(3.110118869878346) q[10];
rz(-3.128235706904837) q[10];
ry(0.037467403383348745) q[11];
rz(-2.0376464895660304) q[11];
ry(-1.5778806731737998) q[12];
rz(-3.1258883008153466) q[12];
ry(-2.029214214234958) q[13];
rz(2.2074233845691955) q[13];
ry(2.5158181674262314) q[14];
rz(-2.684950484484898) q[14];
ry(3.123319549370385) q[15];
rz(-2.2646543753689348) q[15];
ry(-2.5054531531645434) q[16];
rz(1.5620682163423973) q[16];
ry(-2.8356457488504443) q[17];
rz(-1.2690295372829121) q[17];
ry(0.47065207929030356) q[18];
rz(-2.0540681580133144) q[18];
ry(2.464074959279109) q[19];
rz(-2.2692886551286655) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(3.105658233266023) q[0];
rz(0.473166995036677) q[0];
ry(-0.7516755106359095) q[1];
rz(-0.06284289558318734) q[1];
ry(0.4415951437160191) q[2];
rz(-2.4486800233101085) q[2];
ry(-0.9845062750330645) q[3];
rz(-2.8591093901671982) q[3];
ry(-0.4771656907856282) q[4];
rz(-1.7135366998695454) q[4];
ry(-2.753405079559256) q[5];
rz(0.7749514757952163) q[5];
ry(0.17711946848309237) q[6];
rz(-2.0959115684506395) q[6];
ry(0.09671338385948848) q[7];
rz(1.715901634916035) q[7];
ry(1.8949227951063956) q[8];
rz(0.7389404923150495) q[8];
ry(-3.119579803479933) q[9];
rz(1.670208239812701) q[9];
ry(0.37438314397547057) q[10];
rz(1.5142119728933698) q[10];
ry(0.41292253666841733) q[11];
rz(-2.029621638164324) q[11];
ry(0.07479983514039736) q[12];
rz(-0.05920947308198432) q[12];
ry(1.5801381251808306) q[13];
rz(-0.08587273157581264) q[13];
ry(-0.8338661588289035) q[14];
rz(-2.8960789599002426) q[14];
ry(-2.549534091270076) q[15];
rz(0.2968700886513861) q[15];
ry(-1.9985102000145814) q[16];
rz(-0.40133509798103456) q[16];
ry(-0.03871243578766137) q[17];
rz(-0.8222787450334943) q[17];
ry(-0.09424203836369749) q[18];
rz(-0.665903404060633) q[18];
ry(2.2559418352727936) q[19];
rz(0.8444109249123889) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.872505994376411) q[0];
rz(-2.8671870813962665) q[0];
ry(0.8732377846276782) q[1];
rz(-1.441831664811942) q[1];
ry(-0.3619370219952567) q[2];
rz(-0.7295326512780773) q[2];
ry(-0.05161094555404411) q[3];
rz(-2.500768539409122) q[3];
ry(-2.9916105284723575) q[4];
rz(-0.4215127554401547) q[4];
ry(1.7723232052472329) q[5];
rz(1.8238455581568314) q[5];
ry(3.1259226546881043) q[6];
rz(1.7011499911603618) q[6];
ry(0.08460236240615869) q[7];
rz(-0.9637812332672908) q[7];
ry(0.07506613593826664) q[8];
rz(2.600155165443004) q[8];
ry(3.1171498739708277) q[9];
rz(-1.9669472132420935) q[9];
ry(0.7784300893379743) q[10];
rz(-2.700197528947538) q[10];
ry(3.1057911673350715) q[11];
rz(-3.1132707780509823) q[11];
ry(-0.12312320699491153) q[12];
rz(2.619880881054266) q[12];
ry(2.839618744941194) q[13];
rz(3.039254581336511) q[13];
ry(1.562835435417525) q[14];
rz(-0.026331804970157968) q[14];
ry(1.2630146653836825) q[15];
rz(0.024708880241616882) q[15];
ry(1.61223639731902) q[16];
rz(2.436872304084372) q[16];
ry(2.8595714249925086) q[17];
rz(0.026776383795372283) q[17];
ry(-0.5102009560303037) q[18];
rz(2.2639796410923108) q[18];
ry(0.9208851352418462) q[19];
rz(2.4150045016812265) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.21400867170972604) q[0];
rz(-2.7242190291366906) q[0];
ry(1.1001106745853368) q[1];
rz(2.5143488290406735) q[1];
ry(1.3064871542204664) q[2];
rz(1.962577660998933) q[2];
ry(-2.4232702636840644) q[3];
rz(0.3093990539678293) q[3];
ry(1.7140427059268668) q[4];
rz(1.4757856435159813) q[4];
ry(-2.9723580437474535) q[5];
rz(0.5779326772061605) q[5];
ry(-2.9206531075424227) q[6];
rz(-2.2284932680796317) q[6];
ry(-3.079884095949538) q[7];
rz(0.7394598803992202) q[7];
ry(1.2366226968901053) q[8];
rz(-1.285041588325792) q[8];
ry(0.1686977871089077) q[9];
rz(2.9979705908399974) q[9];
ry(-0.12973027507030563) q[10];
rz(-0.5172960969968354) q[10];
ry(-0.06919698839921662) q[11];
rz(-2.6007428727369497) q[11];
ry(-0.0014160387042170945) q[12];
rz(2.6839656088419774) q[12];
ry(-3.0829571696109586) q[13];
rz(-1.5618308246484383) q[13];
ry(-0.1716578487267099) q[14];
rz(-0.8717568581615893) q[14];
ry(-1.5598582914922217) q[15];
rz(-2.521116252621364) q[15];
ry(-0.9986402808428867) q[16];
rz(-2.15612508593809) q[16];
ry(-0.47121195086860485) q[17];
rz(-1.6122656534935587) q[17];
ry(1.7142675418126787) q[18];
rz(-0.6639374650313972) q[18];
ry(1.1480184775847362) q[19];
rz(-2.3633767671729045) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.0271211435769247) q[0];
rz(-0.4000179209460777) q[0];
ry(0.21991672259836353) q[1];
rz(-2.734677734093679) q[1];
ry(0.010857791959116625) q[2];
rz(0.5226706838313806) q[2];
ry(-2.4544820836260857) q[3];
rz(-1.1589940170447193) q[3];
ry(0.17553873662037667) q[4];
rz(2.972788624350433) q[4];
ry(0.12548168519175906) q[5];
rz(1.1872474706154472) q[5];
ry(1.6089619668738617) q[6];
rz(3.0966571558433245) q[6];
ry(0.2621463361196836) q[7];
rz(0.4081662420257155) q[7];
ry(1.1985736456853813) q[8];
rz(-0.8263967608342055) q[8];
ry(2.4529148616422782) q[9];
rz(-0.005639666229699358) q[9];
ry(-2.4930751539439284) q[10];
rz(0.45143339187293297) q[10];
ry(-2.3264766984483907) q[11];
rz(0.9835007814939116) q[11];
ry(1.5455364139930845) q[12];
rz(-2.297957587732759) q[12];
ry(1.4498496769844444) q[13];
rz(-2.986978322954406) q[13];
ry(-0.9676606085618908) q[14];
rz(-2.8007405425292413) q[14];
ry(-0.8615786601317762) q[15];
rz(-1.0366128423877656) q[15];
ry(1.6277606810925942) q[16];
rz(1.7705713837877015) q[16];
ry(-1.2468096718160258) q[17];
rz(-2.1949065375868235) q[17];
ry(-2.1076922775160774) q[18];
rz(-2.0962771844526973) q[18];
ry(-1.543879191115301) q[19];
rz(1.610836979468146) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.18249017166316417) q[0];
rz(-2.456090538840595) q[0];
ry(-2.6273983696998067) q[1];
rz(0.47089104818279404) q[1];
ry(3.0214047833130118) q[2];
rz(-0.9373013064875683) q[2];
ry(-2.0580792467414466) q[3];
rz(-0.6998447435153742) q[3];
ry(-2.7248367986434365) q[4];
rz(-0.45920503662379014) q[4];
ry(1.5723924010574526) q[5];
rz(0.0009156631251763729) q[5];
ry(3.063848558838362) q[6];
rz(-0.6239164866828844) q[6];
ry(-3.0849201569440745) q[7];
rz(2.6195894114692755) q[7];
ry(-0.34124746728619826) q[8];
rz(-1.0699912465681758) q[8];
ry(-0.038440406322147666) q[9];
rz(-0.5096923164968183) q[9];
ry(3.0683062835443233) q[10];
rz(0.9720428670584618) q[10];
ry(0.2448133208307224) q[11];
rz(1.4591514692830077) q[11];
ry(-1.681149441089758) q[12];
rz(0.5256658808642981) q[12];
ry(3.138476523364845) q[13];
rz(3.0057280880465647) q[13];
ry(3.1187445603830724) q[14];
rz(-2.1893216544888405) q[14];
ry(-3.141090733071509) q[15];
rz(1.9425088975060556) q[15];
ry(-0.06328902272673974) q[16];
rz(2.5993754292781066) q[16];
ry(2.669289415979842) q[17];
rz(0.24547710665342587) q[17];
ry(-3.089685260742557) q[18];
rz(1.8839016638809687) q[18];
ry(-2.6713248370623117) q[19];
rz(1.6143178197917096) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.234415208403813) q[0];
rz(-0.3754946872359479) q[0];
ry(0.6756910289755791) q[1];
rz(-1.3852259365051793) q[1];
ry(-3.0282241207031593) q[2];
rz(1.138252930734315) q[2];
ry(-1.1644743786243206) q[3];
rz(0.9746684196358295) q[3];
ry(1.0795635477120231) q[4];
rz(-0.015462808693099735) q[4];
ry(-0.8844579462740382) q[5];
rz(0.8941142262330439) q[5];
ry(-0.02731407627980857) q[6];
rz(-1.5024334945923234) q[6];
ry(-0.19204462903729722) q[7];
rz(0.5987131102139773) q[7];
ry(1.586697024824483) q[8];
rz(0.27579304304966) q[8];
ry(-2.09582808385602) q[9];
rz(-2.180625754213994) q[9];
ry(-2.2141549372931566) q[10];
rz(-0.3464889444103403) q[10];
ry(-3.0930282607874577) q[11];
rz(0.6228011300881628) q[11];
ry(-3.0408330387701623) q[12];
rz(0.11031959185500961) q[12];
ry(0.040379723899140144) q[13];
rz(1.1362642575053334) q[13];
ry(-2.57809473081555) q[14];
rz(3.0171329486768608) q[14];
ry(-2.5026445676661853) q[15];
rz(0.8248889799277997) q[15];
ry(-1.982592214888462) q[16];
rz(-1.481483588521856) q[16];
ry(-1.4670205244136891) q[17];
rz(-2.301829415024538) q[17];
ry(2.64463463082529) q[18];
rz(0.7081330630015357) q[18];
ry(-2.1247455846185654) q[19];
rz(2.728737579762815) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.041684264161874165) q[0];
rz(1.3819884052742661) q[0];
ry(-0.7735121067458568) q[1];
rz(-1.4623070314686581) q[1];
ry(1.9657358856492362) q[2];
rz(-2.977113994530413) q[2];
ry(-1.5596840739781372) q[3];
rz(0.014314250282241309) q[3];
ry(2.8658490325975374) q[4];
rz(0.013393546192393302) q[4];
ry(-3.1412053803620563) q[5];
rz(-2.245436235201105) q[5];
ry(3.1286971463231237) q[6];
rz(0.9662008284851833) q[6];
ry(-3.141128186057403) q[7];
rz(-2.0995197998774295) q[7];
ry(-2.3237352276911447) q[8];
rz(1.7399776707227055) q[8];
ry(0.04326073202961833) q[9];
rz(1.8145182219368143) q[9];
ry(0.0553261054382288) q[10];
rz(2.3614982229403987) q[10];
ry(-1.461374365315711) q[11];
rz(-1.9922032534901764) q[11];
ry(-0.6439734665235415) q[12];
rz(2.5906659461391857) q[12];
ry(0.0009573772885023146) q[13];
rz(3.1409430799807265) q[13];
ry(3.069124313928881) q[14];
rz(-1.6797501152824301) q[14];
ry(-0.038141614327662055) q[15];
rz(-0.9226918737135215) q[15];
ry(2.9396049041652694) q[16];
rz(-1.1010649071336176) q[16];
ry(2.641734262856531) q[17];
rz(-2.813244794318013) q[17];
ry(-2.518736114408986) q[18];
rz(0.053465889453032496) q[18];
ry(-1.186356735884918) q[19];
rz(0.14523545554783132) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.0661089934675734) q[0];
rz(-1.374409758747258) q[0];
ry(-2.586286939962231) q[1];
rz(2.1553836856038933) q[1];
ry(0.1454566313274821) q[2];
rz(-0.009941682433583827) q[2];
ry(-0.6763656914878551) q[3];
rz(-0.2132669312644495) q[3];
ry(-1.5249385352867655) q[4];
rz(-2.1956742395899513) q[4];
ry(-2.3399360159995624) q[5];
rz(0.0033537265427305844) q[5];
ry(-1.3045642650111617) q[6];
rz(-3.139600339387) q[6];
ry(3.0538779144442216) q[7];
rz(2.6891590637741265) q[7];
ry(-3.124637306872245) q[8];
rz(-1.4040603404443983) q[8];
ry(-2.859445660088187) q[9];
rz(-0.7523269809460471) q[9];
ry(-0.04764384442373689) q[10];
rz(-2.6827696403582446) q[10];
ry(0.010651520491473583) q[11];
rz(0.9417609420387486) q[11];
ry(3.134082009764461) q[12];
rz(-1.0123723165677643) q[12];
ry(0.002452637598608476) q[13];
rz(0.46360606673089505) q[13];
ry(2.7321919100279586) q[14];
rz(-1.4826074570720271) q[14];
ry(0.4134757097008208) q[15];
rz(0.12819143864530336) q[15];
ry(-0.27534544864536326) q[16];
rz(0.6799508124038258) q[16];
ry(-2.5410213200539284) q[17];
rz(-0.5999886967804953) q[17];
ry(0.8718096809222805) q[18];
rz(-1.1234957183356498) q[18];
ry(0.6649865819069354) q[19];
rz(1.069518924679635) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.034199984344615686) q[0];
rz(1.5045117655165041) q[0];
ry(-2.9875404588100443) q[1];
rz(-1.079195857268715) q[1];
ry(1.1909917863603297) q[2];
rz(-0.08339721203515107) q[2];
ry(2.6404064040230026) q[3];
rz(1.3645276955349077) q[3];
ry(-0.003431967651249398) q[4];
rz(1.9827853038100463) q[4];
ry(3.037516613753098) q[5];
rz(1.8963383729638932) q[5];
ry(3.0248349277474413) q[6];
rz(-2.573978345425203) q[6];
ry(-3.131477735499218) q[7];
rz(2.7500496381203856) q[7];
ry(-2.1474814514062945) q[8];
rz(2.9412732511971464) q[8];
ry(3.045950205212464) q[9];
rz(-0.7410152289301628) q[9];
ry(0.1602005475712156) q[10];
rz(1.5885803371700522) q[10];
ry(2.006967510152027) q[11];
rz(-2.5591319338905487) q[11];
ry(1.1161468738356466) q[12];
rz(2.1007264020620973) q[12];
ry(-3.1301750046866843) q[13];
rz(1.1821555224882978) q[13];
ry(-0.0002882663269615106) q[14];
rz(-2.3482770684260164) q[14];
ry(3.070660298308234) q[15];
rz(1.4497281926202357) q[15];
ry(-0.30334677710118463) q[16];
rz(0.6799933843125279) q[16];
ry(-2.950924944654754) q[17];
rz(-0.2761698453652101) q[17];
ry(1.53895942666537) q[18];
rz(1.9060104073788322) q[18];
ry(0.49592808191818705) q[19];
rz(2.0714085542027583) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.6743682145756833) q[0];
rz(0.1848722094010672) q[0];
ry(1.4223373295596324) q[1];
rz(-0.9467529686913877) q[1];
ry(-1.705672749151501) q[2];
rz(-1.5697137239230168) q[2];
ry(-1.5727565687471534) q[3];
rz(-0.3410793170930533) q[3];
ry(-0.03781394772137947) q[4];
rz(-1.3436246907012392) q[4];
ry(-0.0932100000912431) q[5];
rz(-1.5184267146885109) q[5];
ry(1.819353376014859) q[6];
rz(1.8768149009118522) q[6];
ry(1.689539913185925) q[7];
rz(-1.688430933613378) q[7];
ry(1.1067183876166071) q[8];
rz(-0.2501268130998069) q[8];
ry(0.6414756498437164) q[9];
rz(-2.8033357948007738) q[9];
ry(-0.03802715242628629) q[10];
rz(-2.387049089134409) q[10];
ry(-3.037929309150126) q[11];
rz(-0.9344515700859308) q[11];
ry(-2.9104684656925937) q[12];
rz(-0.796783463769255) q[12];
ry(-2.0474242421231343) q[13];
rz(-0.7789545564896141) q[13];
ry(-2.5296543647720586) q[14];
rz(2.9731246452630744) q[14];
ry(-2.994350644014708) q[15];
rz(-0.19704544477187322) q[15];
ry(0.9763481305808774) q[16];
rz(2.4631701380249202) q[16];
ry(0.44228508604171296) q[17];
rz(-1.211903845924237) q[17];
ry(-0.22830541104362975) q[18];
rz(-0.816761245400877) q[18];
ry(-0.09941359655954975) q[19];
rz(-2.178555442201062) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.949999650751636) q[0];
rz(2.50466861729806) q[0];
ry(3.129297605159634) q[1];
rz(1.6594885605586507) q[1];
ry(0.2744939268658808) q[2];
rz(3.112741268666724) q[2];
ry(3.1262750855286936) q[3];
rz(-1.5374711776697425) q[3];
ry(-3.1320027254777054) q[4];
rz(-0.025823759230602917) q[4];
ry(-0.023313169016605073) q[5];
rz(1.701351929631966) q[5];
ry(-3.1315263743275215) q[6];
rz(-2.1963953424460794) q[6];
ry(-0.03230425594947839) q[7];
rz(0.7939440527734751) q[7];
ry(-3.131083563958057) q[8];
rz(-3.14102825093998) q[8];
ry(3.009841599657106) q[9];
rz(-1.5600505309256611) q[9];
ry(0.04848467992511964) q[10];
rz(2.129519117496256) q[10];
ry(0.8433027568533644) q[11];
rz(2.367777048749114) q[11];
ry(0.03682348260755841) q[12];
rz(1.312505433311456) q[12];
ry(-0.0001816071299169443) q[13];
rz(-2.3938327168730136) q[13];
ry(-3.0575941950680017) q[14];
rz(1.0401882862869138) q[14];
ry(-3.113698068339634) q[15];
rz(-1.7500424899871607) q[15];
ry(-3.1165843356400242) q[16];
rz(2.107081400514671) q[16];
ry(-2.994517518125821) q[17];
rz(-0.6674647402064249) q[17];
ry(1.423080481993043) q[18];
rz(0.2740606268026076) q[18];
ry(-2.2301358341040434) q[19];
rz(2.02787206921409) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.6682820015688082) q[0];
rz(-1.383633104741655) q[0];
ry(-0.7265364604728912) q[1];
rz(-0.11144219892515395) q[1];
ry(3.063496145358461) q[2];
rz(2.351410353664129) q[2];
ry(0.08081866515050232) q[3];
rz(2.4800844139893576) q[3];
ry(1.7064771362250353) q[4];
rz(3.1081529025550862) q[4];
ry(0.10027693431777897) q[5];
rz(2.6314634645273043) q[5];
ry(-2.381055844707708) q[6];
rz(-2.2520639392673405) q[6];
ry(-0.42050862542173917) q[7];
rz(-0.4172724005925403) q[7];
ry(-2.290522894966015) q[8];
rz(-0.06671670117040782) q[8];
ry(1.2997587129878962) q[9];
rz(-1.0050895114889302) q[9];
ry(2.487922688903481) q[10];
rz(-1.3132277113001782) q[10];
ry(2.556335144320536) q[11];
rz(0.16463222097290145) q[11];
ry(2.2871107938875856) q[12];
rz(-2.350984996815) q[12];
ry(-2.0734615765730813) q[13];
rz(-1.6675028113096397) q[13];
ry(-0.6560935703770826) q[14];
rz(-0.0011986684807521184) q[14];
ry(-0.07280329702607435) q[15];
rz(3.020395966502267) q[15];
ry(0.4985146624510195) q[16];
rz(-2.8878411015949896) q[16];
ry(-1.9216583868263797) q[17];
rz(0.8285045275352544) q[17];
ry(-0.7275148550288684) q[18];
rz(-1.9066586601809554) q[18];
ry(-1.5197812014761762) q[19];
rz(-1.5209995180376448) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.238374531741898) q[0];
rz(-3.007940086459539) q[0];
ry(1.5691519756982917) q[1];
rz(-1.578332896636551) q[1];
ry(-0.0044604313516027885) q[2];
rz(-2.3539054791929424) q[2];
ry(0.029145882661975264) q[3];
rz(-2.1632007048771786) q[3];
ry(-3.0858432463011507) q[4];
rz(-2.8185518009211905) q[4];
ry(-0.05539408971558846) q[5];
rz(-3.129414240819955) q[5];
ry(0.05344585651664789) q[6];
rz(2.7896510214130528) q[6];
ry(-3.1165768081422054) q[7];
rz(-1.5144401809797206) q[7];
ry(-3.1171923024019033) q[8];
rz(3.1091657277955784) q[8];
ry(-0.023031448769948785) q[9];
rz(0.2227270365838372) q[9];
ry(-3.047007658401203) q[10];
rz(-0.31739879502008694) q[10];
ry(2.9151157175500564) q[11];
rz(0.549395296132986) q[11];
ry(0.08262233481536008) q[12];
rz(1.9099405303609247) q[12];
ry(0.00797699335916071) q[13];
rz(-1.5425168947282322) q[13];
ry(3.0836907392703155) q[14];
rz(-1.0993562925399136) q[14];
ry(0.06658926812303799) q[15];
rz(-0.6613986055061369) q[15];
ry(-0.22400730621470366) q[16];
rz(0.14990763552759853) q[16];
ry(3.124675131798545) q[17];
rz(1.9188490152831683) q[17];
ry(2.897045804771836) q[18];
rz(1.2061245622059853) q[18];
ry(1.061044162659993) q[19];
rz(0.4234474474934352) q[19];
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
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.141488479618918) q[0];
rz(0.9939864254733716) q[0];
ry(-1.571667827811666) q[1];
rz(1.9375471706370506) q[1];
ry(1.5883571730585233) q[2];
rz(2.335272902998032) q[2];
ry(-3.080867871570019) q[3];
rz(-0.43515340459271845) q[3];
ry(-0.13171969127347882) q[4];
rz(-3.09205873104129) q[4];
ry(-1.5778399760649582) q[5];
rz(-2.981031945469701) q[5];
ry(-1.5107318433396868) q[6];
rz(-2.8090658218094875) q[6];
ry(3.134711700943687) q[7];
rz(1.8760749944052049) q[7];
ry(1.1600424416413289) q[8];
rz(0.002052092291307872) q[8];
ry(-0.13882413648338154) q[9];
rz(1.1727657466186558) q[9];
ry(-1.792062498403749) q[10];
rz(-1.3154958832210841) q[10];
ry(1.5275136306358776) q[11];
rz(2.0554845910210884) q[11];
ry(1.848862139486017) q[12];
rz(1.0913125612215522) q[12];
ry(0.8971017760271508) q[13];
rz(-2.500697353405597) q[13];
ry(2.005833305291687) q[14];
rz(2.299744799538858) q[14];
ry(1.5492927699122976) q[15];
rz(-0.8774190124397871) q[15];
ry(0.03823060732615761) q[16];
rz(-2.158344844487991) q[16];
ry(-0.7852104561352578) q[17];
rz(-3.104752309817202) q[17];
ry(-0.012408235318792649) q[18];
rz(-2.309762894615624) q[18];
ry(1.487716840662702) q[19];
rz(-2.44839004822425) q[19];