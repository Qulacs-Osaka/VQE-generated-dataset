OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.7519844960775606) q[0];
rz(2.9144051187933524) q[0];
ry(2.6433321269405656) q[1];
rz(-0.19791561619924988) q[1];
ry(-2.013059981335445) q[2];
rz(0.7492417399103265) q[2];
ry(-2.4037114599267566) q[3];
rz(0.2923983797703731) q[3];
ry(0.4397319948217495) q[4];
rz(0.7034906181245862) q[4];
ry(-2.5846775084800764) q[5];
rz(0.6243947801409124) q[5];
ry(-0.46455952140599943) q[6];
rz(-1.0260596880267867) q[6];
ry(0.14252792819444515) q[7];
rz(2.1885386861738105) q[7];
ry(1.5395671786361929) q[8];
rz(1.4557695362826921) q[8];
ry(-0.4331727465603893) q[9];
rz(0.4314258549119911) q[9];
ry(0.5568097288284577) q[10];
rz(-2.6876741870918783) q[10];
ry(-1.2760910286509697) q[11];
rz(-2.6034025389406414) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.0248613990133544) q[0];
rz(-0.48279454022951196) q[0];
ry(-0.5980710875212027) q[1];
rz(-2.80073346149143) q[1];
ry(2.886028116395514) q[2];
rz(1.121026597544742) q[2];
ry(-2.7006759250054304) q[3];
rz(2.854786977246088) q[3];
ry(-1.6670889477104707) q[4];
rz(2.5468340697072693) q[4];
ry(0.16749421333454056) q[5];
rz(2.889219054764198) q[5];
ry(-2.119338921580465) q[6];
rz(-0.9384830693618013) q[6];
ry(0.3006008605684327) q[7];
rz(0.4292594309161834) q[7];
ry(-2.2688550443972035) q[8];
rz(1.3443726421708864) q[8];
ry(-1.201496676737182) q[9];
rz(-1.543723429724248) q[9];
ry(2.718127527161431) q[10];
rz(1.2665254229323801) q[10];
ry(-0.5622137922745442) q[11];
rz(-2.7623312395536574) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.7225871290900407) q[0];
rz(-1.3071767873520321) q[0];
ry(-0.7889690054169585) q[1];
rz(-2.9419680473338) q[1];
ry(1.3246090153543228) q[2];
rz(2.957295818131113) q[2];
ry(1.8802992602789834) q[3];
rz(0.28353876456901217) q[3];
ry(0.432280100581757) q[4];
rz(0.8320658351036467) q[4];
ry(-2.489087458439553) q[5];
rz(-0.03696375212032458) q[5];
ry(0.44620663507481506) q[6];
rz(0.10556108989954538) q[6];
ry(1.3114818910023451) q[7];
rz(0.6104415588319391) q[7];
ry(1.886636394893556) q[8];
rz(0.17283998736229483) q[8];
ry(-2.9780132006994995) q[9];
rz(2.5069073885652644) q[9];
ry(2.6509703175348465) q[10];
rz(-0.15509945646308232) q[10];
ry(0.7543720398628473) q[11];
rz(-1.6678632890375986) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.3756549149475628) q[0];
rz(-2.5029939149456752) q[0];
ry(-0.9284402035450348) q[1];
rz(-1.9549788829752819) q[1];
ry(0.9950361384303399) q[2];
rz(-2.650195995930002) q[2];
ry(1.5794733903934075) q[3];
rz(0.6781948844846907) q[3];
ry(-0.8863415365895625) q[4];
rz(-2.365392262256659) q[4];
ry(0.31359851928369853) q[5];
rz(1.834033120940572) q[5];
ry(0.7467911017476281) q[6];
rz(-1.2865205579985872) q[6];
ry(0.25929282835345724) q[7];
rz(2.542017340620609) q[7];
ry(1.096170644091259) q[8];
rz(0.13457287072637217) q[8];
ry(-2.153060485506368) q[9];
rz(-2.5254899487150073) q[9];
ry(1.0515359079298345) q[10];
rz(0.8888583935932495) q[10];
ry(0.3674171476948933) q[11];
rz(2.187113612877744) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.5789920846919427) q[0];
rz(-1.2487127629660293) q[0];
ry(1.3971843312831052) q[1];
rz(-1.290690302161237) q[1];
ry(1.3506238564783244) q[2];
rz(-1.0588721402132517) q[2];
ry(-0.9170801128268611) q[3];
rz(0.34255275376588745) q[3];
ry(1.5002238595721522) q[4];
rz(-1.0860840407899373) q[4];
ry(0.7645148621818) q[5];
rz(-1.300311816109426) q[5];
ry(2.4382440715525266) q[6];
rz(-1.7683700697109537) q[6];
ry(2.587050002396301) q[7];
rz(1.3771979933594654) q[7];
ry(0.7859710975588982) q[8];
rz(-0.5976717974590463) q[8];
ry(-0.9781605592475211) q[9];
rz(-2.439263211612634) q[9];
ry(2.482594895068669) q[10];
rz(1.2713666626549116) q[10];
ry(0.3036354289081937) q[11];
rz(-1.9122026635173457) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.772012600252316) q[0];
rz(1.7025320322503728) q[0];
ry(-0.635032671112854) q[1];
rz(2.8113903602976738) q[1];
ry(0.5054970327410717) q[2];
rz(0.2844720704645113) q[2];
ry(0.5518707286844459) q[3];
rz(0.42298700344210505) q[3];
ry(-1.3289025502979028) q[4];
rz(2.22491997703915) q[4];
ry(-1.2792377342338057) q[5];
rz(2.4163577309222735) q[5];
ry(1.1079392819262264) q[6];
rz(2.134425567636583) q[6];
ry(-1.62218772926599) q[7];
rz(2.9518532948380733) q[7];
ry(-1.9698448680545564) q[8];
rz(-0.5211994409009648) q[8];
ry(-0.18720581125882504) q[9];
rz(-2.021667968789467) q[9];
ry(-0.3250396813612291) q[10];
rz(-1.3808891393025127) q[10];
ry(0.61078823924411) q[11];
rz(-0.6799445845776882) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.9308002108231221) q[0];
rz(-0.7009003810591763) q[0];
ry(2.521399124460762) q[1];
rz(-1.123223243092572) q[1];
ry(-1.218529955177547) q[2];
rz(2.028545494748999) q[2];
ry(-0.2572148093266335) q[3];
rz(-2.9234118704377523) q[3];
ry(2.824510935086987) q[4];
rz(0.8486798323369119) q[4];
ry(0.2994896366592892) q[5];
rz(-1.8152724985254425) q[5];
ry(-0.5184586122390752) q[6];
rz(2.8517002935874216) q[6];
ry(1.7050203906226558) q[7];
rz(-1.688491414321831) q[7];
ry(0.38990077703801856) q[8];
rz(0.5710123029611465) q[8];
ry(2.043317984223166) q[9];
rz(2.7462812928132627) q[9];
ry(-1.5801616445188627) q[10];
rz(-0.6713691640467534) q[10];
ry(-0.5914702027084608) q[11];
rz(1.419956151133402) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.00859352818347) q[0];
rz(0.6221975633529081) q[0];
ry(1.4109510227989146) q[1];
rz(-1.0909613926454984) q[1];
ry(-0.4913832092370427) q[2];
rz(0.16118533127429302) q[2];
ry(2.772666921387702) q[3];
rz(-2.9455322570205813) q[3];
ry(-3.0028992289883747) q[4];
rz(-2.9065039811617317) q[4];
ry(2.294818273908383) q[5];
rz(0.8889505452555406) q[5];
ry(0.3755060168649474) q[6];
rz(-1.7156817948887246) q[6];
ry(1.4186376714267266) q[7];
rz(-1.5257080511735852) q[7];
ry(2.67948063646385) q[8];
rz(2.3360366605045697) q[8];
ry(-2.7688856599140665) q[9];
rz(2.0624031752829373) q[9];
ry(2.1988441194813992) q[10];
rz(2.8601966447065683) q[10];
ry(0.9786599004225299) q[11];
rz(-1.429224535973355) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.4746173258122042) q[0];
rz(-0.4558525811257822) q[0];
ry(-2.7810651156827957) q[1];
rz(2.1212488365142637) q[1];
ry(-0.3099121714613367) q[2];
rz(0.07152917604628684) q[2];
ry(-0.48378762383347595) q[3];
rz(-0.3586717496446736) q[3];
ry(1.192048651295171) q[4];
rz(0.5500523842425826) q[4];
ry(1.5989199766153968) q[5];
rz(1.6958040041537854) q[5];
ry(0.6116141218631094) q[6];
rz(0.19648353295790746) q[6];
ry(1.194091253171313) q[7];
rz(1.675887170701466) q[7];
ry(-2.357985791376197) q[8];
rz(-2.0103043247000545) q[8];
ry(0.10140252610548531) q[9];
rz(1.4136743222868269) q[9];
ry(1.6185944839624578) q[10];
rz(1.3502648242511404) q[10];
ry(1.1429478060251457) q[11];
rz(1.5622844900551804) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.6042035993601989) q[0];
rz(0.9271463985399108) q[0];
ry(0.5657726601181895) q[1];
rz(0.5218905134978254) q[1];
ry(-1.3103862225196794) q[2];
rz(3.0643244057751655) q[2];
ry(1.0735094105395016) q[3];
rz(2.2591801340602915) q[3];
ry(-2.0329379486890353) q[4];
rz(1.7941325146104576) q[4];
ry(0.5172358028562342) q[5];
rz(1.6902849788210013) q[5];
ry(-1.6568765945279589) q[6];
rz(1.2804057430163123) q[6];
ry(-0.051637229454513184) q[7];
rz(-3.0479333424655706) q[7];
ry(-1.8513759868009494) q[8];
rz(-2.4419939628660847) q[8];
ry(1.8553376017394871) q[9];
rz(-1.6887894856528587) q[9];
ry(0.49504697931727165) q[10];
rz(-2.6636016707264285) q[10];
ry(-0.4406352601311309) q[11];
rz(0.5478307525268324) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.6724315817683622) q[0];
rz(0.673469358573497) q[0];
ry(-0.9093841966474786) q[1];
rz(3.1139127599135685) q[1];
ry(0.09161635067769962) q[2];
rz(-0.08867129966152217) q[2];
ry(-1.847681284373662) q[3];
rz(-1.1870135373990425) q[3];
ry(2.4512002865797413) q[4];
rz(2.671031753850685) q[4];
ry(-2.9103601729063353) q[5];
rz(2.0970719742459223) q[5];
ry(-2.3183828445868526) q[6];
rz(-3.035986221640383) q[6];
ry(-0.7049680751052527) q[7];
rz(-1.3759501400489176) q[7];
ry(-1.7580353446239245) q[8];
rz(-2.605064410875485) q[8];
ry(-1.2885375314919756) q[9];
rz(-2.6816436449475693) q[9];
ry(1.589311049965928) q[10];
rz(-2.632152197990163) q[10];
ry(-2.735930093455437) q[11];
rz(-3.1109880451858203) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.7912086870396273) q[0];
rz(-2.77482413176076) q[0];
ry(-1.5627172246181678) q[1];
rz(1.0662115641705352) q[1];
ry(-1.0060463962548285) q[2];
rz(0.4582108222532045) q[2];
ry(1.143943436693076) q[3];
rz(-1.8169733225486415) q[3];
ry(2.9237203359545996) q[4];
rz(-1.2261051533521596) q[4];
ry(0.6821224836373574) q[5];
rz(-2.5710808307049486) q[5];
ry(-2.3339068887858763) q[6];
rz(-1.2558658201826896) q[6];
ry(0.1104218511513313) q[7];
rz(-0.5262110224562146) q[7];
ry(0.19009559881710647) q[8];
rz(0.7516919614563912) q[8];
ry(0.15779535258883118) q[9];
rz(-1.025364937277777) q[9];
ry(-1.0061177466151747) q[10];
rz(1.3466080349749894) q[10];
ry(1.1529376339538306) q[11];
rz(-0.10128133374569273) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.2148113242478225) q[0];
rz(0.2826467564252702) q[0];
ry(-0.39500533044992275) q[1];
rz(1.1105935817985526) q[1];
ry(-1.130005304472434) q[2];
rz(2.7044492271196305) q[2];
ry(-1.1612407375699174) q[3];
rz(0.8049476900415918) q[3];
ry(-2.348535182518883) q[4];
rz(0.007435781181824197) q[4];
ry(-0.9760449516177021) q[5];
rz(-3.0043100667846003) q[5];
ry(-1.4648395885060377) q[6];
rz(0.3058053334316271) q[6];
ry(1.474442445384713) q[7];
rz(1.1072267203235286) q[7];
ry(1.7787618424732123) q[8];
rz(-1.2755855836153784) q[8];
ry(-2.329146597061625) q[9];
rz(-2.9308891189235924) q[9];
ry(-2.665501762685545) q[10];
rz(-0.16904169244486195) q[10];
ry(-1.4074011435503007) q[11];
rz(-2.9496727574792705) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.3343783374406364) q[0];
rz(2.3588804006385464) q[0];
ry(-1.0865250779129303) q[1];
rz(0.651658009720195) q[1];
ry(1.2760183577868291) q[2];
rz(-1.1370178535428233) q[2];
ry(2.5296632075066294) q[3];
rz(-2.3012797830067524) q[3];
ry(2.520308428389402) q[4];
rz(1.2144591850550324) q[4];
ry(0.6941457109024186) q[5];
rz(1.2548442759872014) q[5];
ry(-0.5036528240981148) q[6];
rz(2.3545947494040593) q[6];
ry(-0.5768930711014518) q[7];
rz(1.3869802822623365) q[7];
ry(1.3064637138860133) q[8];
rz(1.939374624777928) q[8];
ry(-0.5279390455664393) q[9];
rz(0.986675596081746) q[9];
ry(0.3352581602023186) q[10];
rz(-1.1466987728332496) q[10];
ry(-2.021311994726028) q[11];
rz(3.084272556929925) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.8977622322689383) q[0];
rz(2.553794080039224) q[0];
ry(2.6073234510249144) q[1];
rz(1.9077435957653028) q[1];
ry(-2.4410003735926997) q[2];
rz(-1.9467022281428177) q[2];
ry(-2.482266103758247) q[3];
rz(-1.617536187890506) q[3];
ry(-1.926048948542321) q[4];
rz(1.1217484950772476) q[4];
ry(-1.0073191613838055) q[5];
rz(-2.0401159603329644) q[5];
ry(1.730946281306576) q[6];
rz(0.4416713207826718) q[6];
ry(-2.4229022711077484) q[7];
rz(-2.6043348831152335) q[7];
ry(-2.956098498583157) q[8];
rz(-1.2290187810390716) q[8];
ry(0.3257818436799695) q[9];
rz(1.8776159107949075) q[9];
ry(2.8217288434055408) q[10];
rz(-0.8360733280505847) q[10];
ry(-1.4281856101626815) q[11];
rz(-1.8141427480090833) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.6460396720368955) q[0];
rz(-0.23779021075498083) q[0];
ry(-2.6807432872248373) q[1];
rz(0.4727344734728274) q[1];
ry(1.5875263774568658) q[2];
rz(0.4173401740997953) q[2];
ry(1.4284748572910484) q[3];
rz(0.8993147627968874) q[3];
ry(-2.0209062650948795) q[4];
rz(0.0996774101178473) q[4];
ry(-0.78706567396464) q[5];
rz(-2.182621590320635) q[5];
ry(2.817722399053703) q[6];
rz(1.1824870691688754) q[6];
ry(-1.7460865579128255) q[7];
rz(-2.765986326626101) q[7];
ry(0.3084705599783035) q[8];
rz(-0.22914370080554036) q[8];
ry(-2.666448285573739) q[9];
rz(1.3283486507986364) q[9];
ry(2.7525378362598683) q[10];
rz(-1.1626933599474427) q[10];
ry(-0.61381910030964) q[11];
rz(1.6193866193399729) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.5045075599465374) q[0];
rz(2.0805985300807346) q[0];
ry(-2.5337465224092424) q[1];
rz(1.629149869692947) q[1];
ry(-1.2885150981885174) q[2];
rz(-1.6237549136214096) q[2];
ry(0.7461452818880548) q[3];
rz(2.4853933504936205) q[3];
ry(1.1071616957490527) q[4];
rz(0.0557263477909924) q[4];
ry(0.11409348712178663) q[5];
rz(1.1300168616985105) q[5];
ry(1.413310228981933) q[6];
rz(0.222947612188606) q[6];
ry(2.2812287861506517) q[7];
rz(-0.8271416492554895) q[7];
ry(2.2841180972565565) q[8];
rz(0.4027143533202749) q[8];
ry(1.2821071561982882) q[9];
rz(1.675230286817557) q[9];
ry(1.2458991601403477) q[10];
rz(1.5366010223758995) q[10];
ry(-1.8254212406525827) q[11];
rz(1.6056813151632006) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.9204784747249803) q[0];
rz(-0.6085373370820666) q[0];
ry(1.3060739521480642) q[1];
rz(-0.2464459838556499) q[1];
ry(-0.7917817008947408) q[2];
rz(0.2011253607561061) q[2];
ry(2.556270080172412) q[3];
rz(-0.5730919835210768) q[3];
ry(-1.2378216278381498) q[4];
rz(1.8349789093526903) q[4];
ry(-0.6909495074952868) q[5];
rz(-1.2560064544528953) q[5];
ry(-1.6789566936313394) q[6];
rz(-1.1615187635909816) q[6];
ry(0.7402551989422452) q[7];
rz(-2.5004336745897966) q[7];
ry(-2.484082931797598) q[8];
rz(0.4364211631955124) q[8];
ry(1.6339016027183666) q[9];
rz(-2.2131460808315255) q[9];
ry(1.0930269156191734) q[10];
rz(2.6118533821774608) q[10];
ry(-2.6633866188378157) q[11];
rz(1.9466049754869645) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.3657607311435958) q[0];
rz(2.9786528753406096) q[0];
ry(1.7984291593107686) q[1];
rz(-2.682217066993912) q[1];
ry(-2.7085016269429527) q[2];
rz(2.949648731662989) q[2];
ry(-1.0194230191258387) q[3];
rz(-0.5350354029362983) q[3];
ry(-1.8188137459989253) q[4];
rz(1.5173324090228588) q[4];
ry(-1.639178482418279) q[5];
rz(-0.14915191716650075) q[5];
ry(1.054663273744655) q[6];
rz(-1.686493037636419) q[6];
ry(0.4534009863269933) q[7];
rz(-1.9009240007083275) q[7];
ry(2.181963112784107) q[8];
rz(-2.1494251957453407) q[8];
ry(-2.984094366016686) q[9];
rz(-1.0555983790297279) q[9];
ry(-1.3952125512727678) q[10];
rz(2.092137732688654) q[10];
ry(-2.1250280996826607) q[11];
rz(-0.020778664183667618) q[11];