OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(2.6404927414831145) q[0];
ry(2.6682216467769257) q[1];
cx q[0],q[1];
ry(-0.9088605862024778) q[0];
ry(-2.831316307200257) q[1];
cx q[0],q[1];
ry(0.7580157943129577) q[1];
ry(-2.654511333390956) q[2];
cx q[1],q[2];
ry(-0.4405123783204141) q[1];
ry(-2.601487914488641) q[2];
cx q[1],q[2];
ry(1.5231931480094874) q[2];
ry(2.4038680868271403) q[3];
cx q[2],q[3];
ry(-1.775068448566011) q[2];
ry(2.3182474500344874) q[3];
cx q[2],q[3];
ry(-3.047846051200116) q[3];
ry(-0.5297575561783683) q[4];
cx q[3],q[4];
ry(-1.0324320681612447) q[3];
ry(1.3065620377962857) q[4];
cx q[3],q[4];
ry(1.3180053501873212) q[4];
ry(-2.8592167785939284) q[5];
cx q[4],q[5];
ry(1.7071305345219785) q[4];
ry(1.3889059291751966) q[5];
cx q[4],q[5];
ry(1.3916652878837477) q[5];
ry(2.496809098128908) q[6];
cx q[5],q[6];
ry(-2.7714703791666078) q[5];
ry(1.683724877983882) q[6];
cx q[5],q[6];
ry(1.6602757232346441) q[6];
ry(-2.420549217404205) q[7];
cx q[6],q[7];
ry(-1.2749778578949529) q[6];
ry(-2.764219752221103) q[7];
cx q[6],q[7];
ry(2.153945388070765) q[0];
ry(2.7692063528658917) q[1];
cx q[0],q[1];
ry(-0.05428832722194547) q[0];
ry(2.802429416086661) q[1];
cx q[0],q[1];
ry(0.39921622011538815) q[1];
ry(-1.539056315046929) q[2];
cx q[1],q[2];
ry(2.8087646414511305) q[1];
ry(-1.217009961658503) q[2];
cx q[1],q[2];
ry(-1.2986827466471782) q[2];
ry(1.7157524646021676) q[3];
cx q[2],q[3];
ry(2.9070556218509607) q[2];
ry(-1.4899806942294185) q[3];
cx q[2],q[3];
ry(-1.8026324358344954) q[3];
ry(0.5660030260022721) q[4];
cx q[3],q[4];
ry(-0.22225761458068405) q[3];
ry(-0.3920961931316551) q[4];
cx q[3],q[4];
ry(0.46714700080278776) q[4];
ry(-1.64093560050234) q[5];
cx q[4],q[5];
ry(1.9286929164070479) q[4];
ry(-2.332634189907931) q[5];
cx q[4],q[5];
ry(2.895920160919301) q[5];
ry(-3.1080172356141627) q[6];
cx q[5],q[6];
ry(2.1506649121501056) q[5];
ry(0.39996893586299465) q[6];
cx q[5],q[6];
ry(-0.28247089259479724) q[6];
ry(-2.1507190983360287) q[7];
cx q[6],q[7];
ry(-2.4766245523675208) q[6];
ry(1.548677344117028) q[7];
cx q[6],q[7];
ry(-1.4895956964300967) q[0];
ry(1.4659005667219809) q[1];
cx q[0],q[1];
ry(3.077354777401409) q[0];
ry(0.829571707585305) q[1];
cx q[0],q[1];
ry(-3.1336873163381638) q[1];
ry(-0.5061992364991799) q[2];
cx q[1],q[2];
ry(-0.5384519682404579) q[1];
ry(-1.4560319926280918) q[2];
cx q[1],q[2];
ry(2.023891312166238) q[2];
ry(-1.3934742136197185) q[3];
cx q[2],q[3];
ry(-1.081627462930638) q[2];
ry(-0.1278006570768042) q[3];
cx q[2],q[3];
ry(0.2462372165754142) q[3];
ry(-1.7417610341949434) q[4];
cx q[3],q[4];
ry(-0.4513693251816064) q[3];
ry(-0.19796903894410728) q[4];
cx q[3],q[4];
ry(0.29180389869234724) q[4];
ry(0.4675658206082048) q[5];
cx q[4],q[5];
ry(1.2874241035668295) q[4];
ry(2.3648452531522253) q[5];
cx q[4],q[5];
ry(-0.11239894656685845) q[5];
ry(-2.8516561926636204) q[6];
cx q[5],q[6];
ry(0.577699849109659) q[5];
ry(2.374743922667806) q[6];
cx q[5],q[6];
ry(-1.7905731254937995) q[6];
ry(0.3955272597382402) q[7];
cx q[6],q[7];
ry(-2.3280582245580472) q[6];
ry(1.8451487718246715) q[7];
cx q[6],q[7];
ry(-1.8011829137209807) q[0];
ry(0.6312608953905393) q[1];
cx q[0],q[1];
ry(0.22310715034509634) q[0];
ry(1.3379877970993652) q[1];
cx q[0],q[1];
ry(2.367449784630363) q[1];
ry(2.0869850383705053) q[2];
cx q[1],q[2];
ry(1.3375076210797878) q[1];
ry(-2.1349833640640234) q[2];
cx q[1],q[2];
ry(-2.499239239930778) q[2];
ry(-0.23151860820990308) q[3];
cx q[2],q[3];
ry(1.0133736863592686) q[2];
ry(2.812829704984536) q[3];
cx q[2],q[3];
ry(0.8208859339586514) q[3];
ry(1.9442642831808108) q[4];
cx q[3],q[4];
ry(0.07470811666323485) q[3];
ry(1.5752775521398454) q[4];
cx q[3],q[4];
ry(-3.128068717430669) q[4];
ry(2.750612785717933) q[5];
cx q[4],q[5];
ry(2.9263518345415864) q[4];
ry(1.8539941882675655) q[5];
cx q[4],q[5];
ry(-2.9764979531087326) q[5];
ry(-0.13236048487342694) q[6];
cx q[5],q[6];
ry(-2.1506650873350734) q[5];
ry(-2.7993502394291503) q[6];
cx q[5],q[6];
ry(0.8370358771272288) q[6];
ry(-2.2748242521905357) q[7];
cx q[6],q[7];
ry(-1.8415428822966324) q[6];
ry(-2.9895841785768424) q[7];
cx q[6],q[7];
ry(2.4750207377702207) q[0];
ry(-0.3075167005425668) q[1];
cx q[0],q[1];
ry(0.4885360056867061) q[0];
ry(-0.35323486231511075) q[1];
cx q[0],q[1];
ry(-0.5513964263643224) q[1];
ry(0.6348790778036344) q[2];
cx q[1],q[2];
ry(1.7673347901261138) q[1];
ry(-1.9620222244336896) q[2];
cx q[1],q[2];
ry(-1.5776993877114287) q[2];
ry(-1.913999250436326) q[3];
cx q[2],q[3];
ry(2.5210531495823503) q[2];
ry(-0.18504694257726792) q[3];
cx q[2],q[3];
ry(1.478786187971642) q[3];
ry(-2.2439045750050166) q[4];
cx q[3],q[4];
ry(0.4279195947015229) q[3];
ry(0.8943439401529906) q[4];
cx q[3],q[4];
ry(0.20943385240759937) q[4];
ry(2.7446276617848255) q[5];
cx q[4],q[5];
ry(-2.6105210814645545) q[4];
ry(-3.0339709695599195) q[5];
cx q[4],q[5];
ry(0.28299803070579127) q[5];
ry(0.9939959437761138) q[6];
cx q[5],q[6];
ry(0.7482583963608936) q[5];
ry(-1.1622923935308647) q[6];
cx q[5],q[6];
ry(-0.3096321003753806) q[6];
ry(3.068520401917502) q[7];
cx q[6],q[7];
ry(-0.132614481347477) q[6];
ry(0.10954266334538154) q[7];
cx q[6],q[7];
ry(1.3749791504590423) q[0];
ry(2.9196936111813607) q[1];
cx q[0],q[1];
ry(-0.8799639520913551) q[0];
ry(0.753761957415737) q[1];
cx q[0],q[1];
ry(-1.3203069898798656) q[1];
ry(-0.6508548673067103) q[2];
cx q[1],q[2];
ry(-1.0582836397575515) q[1];
ry(-2.597262436982759) q[2];
cx q[1],q[2];
ry(-2.2209010080100366) q[2];
ry(0.3030662001875858) q[3];
cx q[2],q[3];
ry(-2.5537850751552984) q[2];
ry(1.6609974746038887) q[3];
cx q[2],q[3];
ry(-2.884395646672917) q[3];
ry(1.7678234374186377) q[4];
cx q[3],q[4];
ry(2.0565295485374873) q[3];
ry(-2.8773590544769454) q[4];
cx q[3],q[4];
ry(-0.18295713795288115) q[4];
ry(0.9946299797784794) q[5];
cx q[4],q[5];
ry(1.2118429703563285) q[4];
ry(-1.9367755954926973) q[5];
cx q[4],q[5];
ry(-2.584784198942487) q[5];
ry(0.9213745065234987) q[6];
cx q[5],q[6];
ry(0.15809035196038554) q[5];
ry(1.7654342939768068) q[6];
cx q[5],q[6];
ry(1.9518746521872457) q[6];
ry(-1.2131546384635197) q[7];
cx q[6],q[7];
ry(2.664628822822619) q[6];
ry(-3.1141198682650244) q[7];
cx q[6],q[7];
ry(-2.020302461472605) q[0];
ry(-0.985262224064579) q[1];
cx q[0],q[1];
ry(-1.0879670157330878) q[0];
ry(2.867383913668795) q[1];
cx q[0],q[1];
ry(1.525592489975474) q[1];
ry(-2.3446020153847686) q[2];
cx q[1],q[2];
ry(-1.7546457281551044) q[1];
ry(2.521455383621098) q[2];
cx q[1],q[2];
ry(0.9868506596929896) q[2];
ry(-0.31030966614012456) q[3];
cx q[2],q[3];
ry(0.4343136507947778) q[2];
ry(0.4054911823285714) q[3];
cx q[2],q[3];
ry(2.294716032956597) q[3];
ry(0.944863919580056) q[4];
cx q[3],q[4];
ry(0.7218724657600768) q[3];
ry(-2.8847752794474313) q[4];
cx q[3],q[4];
ry(1.4655861727059971) q[4];
ry(-0.4450917073232492) q[5];
cx q[4],q[5];
ry(-2.5221058573385786) q[4];
ry(-0.2448205860212506) q[5];
cx q[4],q[5];
ry(2.264377096672823) q[5];
ry(1.9013188185230012) q[6];
cx q[5],q[6];
ry(-0.4004709497493657) q[5];
ry(0.21265806997932973) q[6];
cx q[5],q[6];
ry(1.653948250485815) q[6];
ry(-2.049245854478439) q[7];
cx q[6],q[7];
ry(1.7787247403714819) q[6];
ry(0.9512681116852119) q[7];
cx q[6],q[7];
ry(-2.7267243927334133) q[0];
ry(1.2135776812367653) q[1];
cx q[0],q[1];
ry(1.1232697105675358) q[0];
ry(-2.833501569371764) q[1];
cx q[0],q[1];
ry(2.9016522070241373) q[1];
ry(-2.8435508909023546) q[2];
cx q[1],q[2];
ry(0.13789348633232462) q[1];
ry(1.5940359789153238) q[2];
cx q[1],q[2];
ry(-0.6851918114443487) q[2];
ry(-2.8174602417960277) q[3];
cx q[2],q[3];
ry(-0.5626366579708613) q[2];
ry(-2.846080799814254) q[3];
cx q[2],q[3];
ry(1.9430466240873687) q[3];
ry(-1.7084765261335035) q[4];
cx q[3],q[4];
ry(-2.0459816760834473) q[3];
ry(2.532303736064649) q[4];
cx q[3],q[4];
ry(1.8874854625194837) q[4];
ry(0.4356128404659518) q[5];
cx q[4],q[5];
ry(0.3090409723858256) q[4];
ry(3.0735180645154823) q[5];
cx q[4],q[5];
ry(-3.131268581014461) q[5];
ry(-0.5610996178462812) q[6];
cx q[5],q[6];
ry(0.06892959866245116) q[5];
ry(-1.4970181117361439) q[6];
cx q[5],q[6];
ry(1.6795981448690651) q[6];
ry(0.9295262733125927) q[7];
cx q[6],q[7];
ry(-1.188002876213273) q[6];
ry(-1.2411893186152494) q[7];
cx q[6],q[7];
ry(-1.1608301479706915) q[0];
ry(-1.3258776991219627) q[1];
cx q[0],q[1];
ry(-1.7629756483689643) q[0];
ry(-1.4712142388297489) q[1];
cx q[0],q[1];
ry(1.307827136980854) q[1];
ry(-2.9998149533963336) q[2];
cx q[1],q[2];
ry(3.034614647400247) q[1];
ry(-0.3231620881435547) q[2];
cx q[1],q[2];
ry(2.5849084516638605) q[2];
ry(-0.8527216194130804) q[3];
cx q[2],q[3];
ry(1.2927086050545378) q[2];
ry(-2.56946418554086) q[3];
cx q[2],q[3];
ry(-2.716840410443035) q[3];
ry(2.2363667196258783) q[4];
cx q[3],q[4];
ry(0.581523055036441) q[3];
ry(-2.303978899473565) q[4];
cx q[3],q[4];
ry(1.4847500918056242) q[4];
ry(-1.0521617839959063) q[5];
cx q[4],q[5];
ry(-1.0248463589496684) q[4];
ry(-0.09958886427208835) q[5];
cx q[4],q[5];
ry(-1.5857917141470521) q[5];
ry(1.1529160660091362) q[6];
cx q[5],q[6];
ry(0.4774008745044926) q[5];
ry(0.41390664714123027) q[6];
cx q[5],q[6];
ry(2.9280979572274353) q[6];
ry(-0.8452872270433988) q[7];
cx q[6],q[7];
ry(3.029598737090406) q[6];
ry(2.586542755068954) q[7];
cx q[6],q[7];
ry(-2.22815220667209) q[0];
ry(-2.0617992181828813) q[1];
cx q[0],q[1];
ry(-2.872989315041986) q[0];
ry(-2.42091556518929) q[1];
cx q[0],q[1];
ry(-1.3821004847158045) q[1];
ry(-0.26333673078387) q[2];
cx q[1],q[2];
ry(3.0845954823154997) q[1];
ry(2.774283209007762) q[2];
cx q[1],q[2];
ry(2.918115230458564) q[2];
ry(1.5357520208496709) q[3];
cx q[2],q[3];
ry(2.7764835425202086) q[2];
ry(-2.6941053262260968) q[3];
cx q[2],q[3];
ry(1.1691295386491456) q[3];
ry(-1.7442533350028075) q[4];
cx q[3],q[4];
ry(-0.18486922141625475) q[3];
ry(1.9756497193658902) q[4];
cx q[3],q[4];
ry(-1.5046003696432302) q[4];
ry(-1.2070490103962648) q[5];
cx q[4],q[5];
ry(-2.624214702889723) q[4];
ry(0.18684641419775644) q[5];
cx q[4],q[5];
ry(-1.1328489174871281) q[5];
ry(-1.3916326751491965) q[6];
cx q[5],q[6];
ry(3.1047579787421653) q[5];
ry(-1.2479406013867924) q[6];
cx q[5],q[6];
ry(2.290432332639663) q[6];
ry(1.1001788295553867) q[7];
cx q[6],q[7];
ry(-1.441970192446766) q[6];
ry(0.860656656720793) q[7];
cx q[6],q[7];
ry(-1.12431792745408) q[0];
ry(-2.514389046109677) q[1];
cx q[0],q[1];
ry(1.7273465090032163) q[0];
ry(-2.6183889582767574) q[1];
cx q[0],q[1];
ry(-0.42901977122939833) q[1];
ry(2.8575023669541326) q[2];
cx q[1],q[2];
ry(-2.898457609834112) q[1];
ry(1.2550337460673529) q[2];
cx q[1],q[2];
ry(0.007297757100802931) q[2];
ry(2.1992678479415826) q[3];
cx q[2],q[3];
ry(-1.663283713955222) q[2];
ry(2.0888493862503665) q[3];
cx q[2],q[3];
ry(-0.40377513443651153) q[3];
ry(-0.27365750099481917) q[4];
cx q[3],q[4];
ry(-2.6433266951849292) q[3];
ry(-1.151928317338447) q[4];
cx q[3],q[4];
ry(2.874790793086811) q[4];
ry(-0.246375021219601) q[5];
cx q[4],q[5];
ry(2.7974816569430776) q[4];
ry(0.9300026962827257) q[5];
cx q[4],q[5];
ry(-0.3419831585922959) q[5];
ry(0.2145691517639614) q[6];
cx q[5],q[6];
ry(-0.677985597967548) q[5];
ry(0.4149766022690805) q[6];
cx q[5],q[6];
ry(-0.1866401899781529) q[6];
ry(-2.803223848644694) q[7];
cx q[6],q[7];
ry(2.1539771850650364) q[6];
ry(-2.568886661039828) q[7];
cx q[6],q[7];
ry(1.8194353854809906) q[0];
ry(2.496484976772329) q[1];
cx q[0],q[1];
ry(1.8969941576218237) q[0];
ry(-2.395250339792104) q[1];
cx q[0],q[1];
ry(-2.8012875612062813) q[1];
ry(0.8460812483820388) q[2];
cx q[1],q[2];
ry(1.2035260954681002) q[1];
ry(0.1549439382199722) q[2];
cx q[1],q[2];
ry(1.8787849470537425) q[2];
ry(0.9492484063198955) q[3];
cx q[2],q[3];
ry(1.1251177617647226) q[2];
ry(-2.3852120743498197) q[3];
cx q[2],q[3];
ry(2.3539451116189785) q[3];
ry(-1.4578288675515774) q[4];
cx q[3],q[4];
ry(0.1306053241894806) q[3];
ry(0.5351177510222929) q[4];
cx q[3],q[4];
ry(2.970420075189808) q[4];
ry(-0.05683212021459963) q[5];
cx q[4],q[5];
ry(2.031082505106278) q[4];
ry(2.7911971599931635) q[5];
cx q[4],q[5];
ry(2.0877833007571565) q[5];
ry(2.810404836720905) q[6];
cx q[5],q[6];
ry(-2.1425886566948193) q[5];
ry(-2.458372147330034) q[6];
cx q[5],q[6];
ry(1.2957911442801702) q[6];
ry(-2.0481548944801498) q[7];
cx q[6],q[7];
ry(0.1446537367568341) q[6];
ry(2.29044898086837) q[7];
cx q[6],q[7];
ry(-0.5912577601452513) q[0];
ry(-0.6936880520100033) q[1];
cx q[0],q[1];
ry(0.6909408284237731) q[0];
ry(-0.23828303946582885) q[1];
cx q[0],q[1];
ry(-2.558801005502238) q[1];
ry(-2.217413294728012) q[2];
cx q[1],q[2];
ry(-1.8988890873629913) q[1];
ry(-0.9592446907791086) q[2];
cx q[1],q[2];
ry(-0.9402978470002534) q[2];
ry(-0.33548105642925374) q[3];
cx q[2],q[3];
ry(1.9169714673524958) q[2];
ry(1.7164820879925025) q[3];
cx q[2],q[3];
ry(-0.9646381055143688) q[3];
ry(1.4941579401533869) q[4];
cx q[3],q[4];
ry(1.1987092958750463) q[3];
ry(0.36922802165476476) q[4];
cx q[3],q[4];
ry(-3.1346076858450806) q[4];
ry(-1.0232083216088728) q[5];
cx q[4],q[5];
ry(-0.8145816953169742) q[4];
ry(2.7346733781524892) q[5];
cx q[4],q[5];
ry(2.293713124273692) q[5];
ry(0.7477422850897666) q[6];
cx q[5],q[6];
ry(0.44134977218461585) q[5];
ry(2.779978379225953) q[6];
cx q[5],q[6];
ry(-1.0497494285448141) q[6];
ry(2.5590327587602633) q[7];
cx q[6],q[7];
ry(1.9611535853289432) q[6];
ry(2.012038111319698) q[7];
cx q[6],q[7];
ry(-0.8449509192312741) q[0];
ry(0.7575289459184162) q[1];
cx q[0],q[1];
ry(-1.3481025228068153) q[0];
ry(0.9331563135435719) q[1];
cx q[0],q[1];
ry(-1.8327693974519152) q[1];
ry(-0.6398490703938045) q[2];
cx q[1],q[2];
ry(-1.383190267221564) q[1];
ry(-1.555955909652421) q[2];
cx q[1],q[2];
ry(-0.5867487268996721) q[2];
ry(2.1492907782721407) q[3];
cx q[2],q[3];
ry(-0.4000375879541498) q[2];
ry(3.136762922229844) q[3];
cx q[2],q[3];
ry(-2.7633976166408623) q[3];
ry(1.499148482679427) q[4];
cx q[3],q[4];
ry(-0.6095568810884293) q[3];
ry(-2.451203736900843) q[4];
cx q[3],q[4];
ry(-2.436270002778913) q[4];
ry(-1.5463904581746224) q[5];
cx q[4],q[5];
ry(-1.2258096741624376) q[4];
ry(0.5288848033804252) q[5];
cx q[4],q[5];
ry(-1.530862724578895) q[5];
ry(1.292287089568349) q[6];
cx q[5],q[6];
ry(-0.33318107861421176) q[5];
ry(1.0033973167276748) q[6];
cx q[5],q[6];
ry(-0.5701391996304288) q[6];
ry(1.2333361388283874) q[7];
cx q[6],q[7];
ry(1.8525193868613359) q[6];
ry(-0.645342300124246) q[7];
cx q[6],q[7];
ry(2.922955107856309) q[0];
ry(-1.1607391187199747) q[1];
cx q[0],q[1];
ry(0.9454395367408729) q[0];
ry(0.7836625442615174) q[1];
cx q[0],q[1];
ry(-0.2772941444792369) q[1];
ry(-0.11870942216038394) q[2];
cx q[1],q[2];
ry(2.132179446079295) q[1];
ry(-2.767274507012959) q[2];
cx q[1],q[2];
ry(-0.03365560979419312) q[2];
ry(-0.6702372276066695) q[3];
cx q[2],q[3];
ry(2.312683420367028) q[2];
ry(2.5370863694946855) q[3];
cx q[2],q[3];
ry(1.9461376358600522) q[3];
ry(-1.0577105985805666) q[4];
cx q[3],q[4];
ry(2.0757481965850415) q[3];
ry(1.333533733349431) q[4];
cx q[3],q[4];
ry(2.8610851975696363) q[4];
ry(1.8750867406418585) q[5];
cx q[4],q[5];
ry(0.9940481005618053) q[4];
ry(0.15599905063962072) q[5];
cx q[4],q[5];
ry(-0.15420075777009656) q[5];
ry(-2.133887652528961) q[6];
cx q[5],q[6];
ry(-1.7692963284720502) q[5];
ry(-0.2511204753953793) q[6];
cx q[5],q[6];
ry(-1.0330086461681527) q[6];
ry(-1.5716098934792446) q[7];
cx q[6],q[7];
ry(1.5947323182239135) q[6];
ry(1.6297905440189266) q[7];
cx q[6],q[7];
ry(-2.639909520403665) q[0];
ry(2.4660386876402067) q[1];
cx q[0],q[1];
ry(-2.338631298193948) q[0];
ry(-1.0491096578560224) q[1];
cx q[0],q[1];
ry(1.3617513208646603) q[1];
ry(-1.1991223817345915) q[2];
cx q[1],q[2];
ry(0.9229868584605994) q[1];
ry(-0.390999649223838) q[2];
cx q[1],q[2];
ry(2.3888986033779167) q[2];
ry(-2.2977426959553826) q[3];
cx q[2],q[3];
ry(1.0281638547245653) q[2];
ry(-1.0178351242522738) q[3];
cx q[2],q[3];
ry(-0.8490110276732379) q[3];
ry(2.75356798685901) q[4];
cx q[3],q[4];
ry(0.610946156644987) q[3];
ry(0.6852357551265043) q[4];
cx q[3],q[4];
ry(1.1511837041680506) q[4];
ry(-3.089385507796946) q[5];
cx q[4],q[5];
ry(-0.6082661693674302) q[4];
ry(0.4438505678961354) q[5];
cx q[4],q[5];
ry(-1.457351624576375) q[5];
ry(-2.323632145713376) q[6];
cx q[5],q[6];
ry(0.832409780693046) q[5];
ry(-0.5939806335236303) q[6];
cx q[5],q[6];
ry(-2.261976628545393) q[6];
ry(0.6907354981895107) q[7];
cx q[6],q[7];
ry(1.702983374032355) q[6];
ry(0.163401677425294) q[7];
cx q[6],q[7];
ry(-0.23640818952043968) q[0];
ry(-0.6395410573800042) q[1];
cx q[0],q[1];
ry(0.13727046912951446) q[0];
ry(1.3834566987375352) q[1];
cx q[0],q[1];
ry(-0.2420220152590147) q[1];
ry(-2.6330990949268) q[2];
cx q[1],q[2];
ry(-2.2001044364185045) q[1];
ry(0.7249102661260354) q[2];
cx q[1],q[2];
ry(2.82730553558728) q[2];
ry(2.8023789098000975) q[3];
cx q[2],q[3];
ry(-0.2970884790270452) q[2];
ry(-2.0707300137226294) q[3];
cx q[2],q[3];
ry(-0.9463655474461241) q[3];
ry(-2.869510783173048) q[4];
cx q[3],q[4];
ry(-0.7168188515153773) q[3];
ry(-1.6483651605451468) q[4];
cx q[3],q[4];
ry(-1.3129309871378871) q[4];
ry(-1.3699455721676) q[5];
cx q[4],q[5];
ry(2.4817129155068525) q[4];
ry(-1.054923768767262) q[5];
cx q[4],q[5];
ry(-2.5037315597424867) q[5];
ry(2.40485562946676) q[6];
cx q[5],q[6];
ry(0.9146553111046016) q[5];
ry(2.92097497116933) q[6];
cx q[5],q[6];
ry(2.3825542590809876) q[6];
ry(1.317687098189412) q[7];
cx q[6],q[7];
ry(-3.119668603882264) q[6];
ry(2.218793136414461) q[7];
cx q[6],q[7];
ry(2.087196244492615) q[0];
ry(-1.5430974676776597) q[1];
cx q[0],q[1];
ry(0.6435565403292612) q[0];
ry(2.912919809480102) q[1];
cx q[0],q[1];
ry(-2.691460008143376) q[1];
ry(1.2445869024126708) q[2];
cx q[1],q[2];
ry(0.26870430280516094) q[1];
ry(0.5577169241712586) q[2];
cx q[1],q[2];
ry(2.427275563356775) q[2];
ry(-1.2658672694384352) q[3];
cx q[2],q[3];
ry(1.9012087611391022) q[2];
ry(1.743300103534212) q[3];
cx q[2],q[3];
ry(2.4485450071562305) q[3];
ry(-2.22785617001914) q[4];
cx q[3],q[4];
ry(-2.2824563365918067) q[3];
ry(-0.408924189309821) q[4];
cx q[3],q[4];
ry(-2.118173596888259) q[4];
ry(2.6741838691895388) q[5];
cx q[4],q[5];
ry(-1.2662595061365893) q[4];
ry(-0.22526852080548124) q[5];
cx q[4],q[5];
ry(2.294362596173263) q[5];
ry(-1.0603404398799154) q[6];
cx q[5],q[6];
ry(-1.6023123779168849) q[5];
ry(1.0963954770930993) q[6];
cx q[5],q[6];
ry(1.2950897017873473) q[6];
ry(0.6426672986720972) q[7];
cx q[6],q[7];
ry(1.4370346063371606) q[6];
ry(2.8828299868180913) q[7];
cx q[6],q[7];
ry(2.4362752802512624) q[0];
ry(0.18213915738007191) q[1];
cx q[0],q[1];
ry(0.7227478668484116) q[0];
ry(-2.1340119756361817) q[1];
cx q[0],q[1];
ry(-2.5618821109965952) q[1];
ry(-2.5883047039667684) q[2];
cx q[1],q[2];
ry(-2.664473428158885) q[1];
ry(0.7951915800322614) q[2];
cx q[1],q[2];
ry(-0.8098019111066427) q[2];
ry(-0.11508785514409252) q[3];
cx q[2],q[3];
ry(1.3549449275865104) q[2];
ry(2.110172402904058) q[3];
cx q[2],q[3];
ry(-3.1257771701652723) q[3];
ry(-1.278051341524231) q[4];
cx q[3],q[4];
ry(1.9550922734975478) q[3];
ry(-0.36898180477625075) q[4];
cx q[3],q[4];
ry(2.917868756755067) q[4];
ry(-1.2313675692657449) q[5];
cx q[4],q[5];
ry(2.857644569706356) q[4];
ry(-2.932230606215386) q[5];
cx q[4],q[5];
ry(2.6373367807956583) q[5];
ry(1.8577262612443608) q[6];
cx q[5],q[6];
ry(0.043376031064019216) q[5];
ry(1.4238602655937846) q[6];
cx q[5],q[6];
ry(1.3413962599640046) q[6];
ry(-2.495797247231775) q[7];
cx q[6],q[7];
ry(1.5915299719244969) q[6];
ry(-0.9468194807410674) q[7];
cx q[6],q[7];
ry(0.395307278612602) q[0];
ry(0.9839305546616599) q[1];
cx q[0],q[1];
ry(-2.2266163802285677) q[0];
ry(2.8316760509191687) q[1];
cx q[0],q[1];
ry(1.9709353468498048) q[1];
ry(1.1745620476042582) q[2];
cx q[1],q[2];
ry(-1.4271609840349166) q[1];
ry(1.8150818893420952) q[2];
cx q[1],q[2];
ry(-1.5989151039016871) q[2];
ry(1.6363634169709214) q[3];
cx q[2],q[3];
ry(1.6157595062169192) q[2];
ry(-0.17547911869721666) q[3];
cx q[2],q[3];
ry(2.087700575274182) q[3];
ry(-0.5863124449601429) q[4];
cx q[3],q[4];
ry(-0.8557971180409195) q[3];
ry(-0.5325539431299514) q[4];
cx q[3],q[4];
ry(2.0031138575692697) q[4];
ry(-1.5434973370593763) q[5];
cx q[4],q[5];
ry(3.117392170642684) q[4];
ry(-0.47300622278898535) q[5];
cx q[4],q[5];
ry(-3.0881446516638604) q[5];
ry(1.7338112551657883) q[6];
cx q[5],q[6];
ry(-2.090124438593126) q[5];
ry(0.6060618984153441) q[6];
cx q[5],q[6];
ry(-2.9309856859812213) q[6];
ry(0.17956684410021762) q[7];
cx q[6],q[7];
ry(0.9183594187304669) q[6];
ry(-2.4770983794108927) q[7];
cx q[6],q[7];
ry(1.0502032083517925) q[0];
ry(-2.1559371140665453) q[1];
cx q[0],q[1];
ry(1.0033175134961119) q[0];
ry(-0.3541904483647479) q[1];
cx q[0],q[1];
ry(0.18706773353080672) q[1];
ry(-1.4553215042166858) q[2];
cx q[1],q[2];
ry(-0.7751201268502541) q[1];
ry(-3.107920420283853) q[2];
cx q[1],q[2];
ry(2.175855357554713) q[2];
ry(2.3448870601582037) q[3];
cx q[2],q[3];
ry(-2.0392961172239525) q[2];
ry(1.21705102471049) q[3];
cx q[2],q[3];
ry(-1.1566006471726045) q[3];
ry(0.08854778497239896) q[4];
cx q[3],q[4];
ry(-1.0320925279406499) q[3];
ry(1.1486744807542917) q[4];
cx q[3],q[4];
ry(-1.1116288748300673) q[4];
ry(0.8076724393922277) q[5];
cx q[4],q[5];
ry(2.508821937836273) q[4];
ry(-0.5327715814823062) q[5];
cx q[4],q[5];
ry(-2.1525357424433826) q[5];
ry(-1.7034718168067853) q[6];
cx q[5],q[6];
ry(-1.5174521688808316) q[5];
ry(-0.01373970207604465) q[6];
cx q[5],q[6];
ry(2.4590136717815914) q[6];
ry(-1.9717235768742878) q[7];
cx q[6],q[7];
ry(1.2102934824254952) q[6];
ry(0.4820045180348318) q[7];
cx q[6],q[7];
ry(-0.841977071481927) q[0];
ry(2.3163492596076027) q[1];
cx q[0],q[1];
ry(2.1710057870177155) q[0];
ry(-3.1382424207709647) q[1];
cx q[0],q[1];
ry(0.6731020440057993) q[1];
ry(2.580667484241254) q[2];
cx q[1],q[2];
ry(1.1402151700351713) q[1];
ry(1.6625432592818132) q[2];
cx q[1],q[2];
ry(0.813790179400117) q[2];
ry(-1.2484561603359599) q[3];
cx q[2],q[3];
ry(0.3974769265564247) q[2];
ry(2.4184617400235044) q[3];
cx q[2],q[3];
ry(-2.121819474275636) q[3];
ry(-0.18339129117580466) q[4];
cx q[3],q[4];
ry(0.049682355762487695) q[3];
ry(-0.275688983991167) q[4];
cx q[3],q[4];
ry(-2.619426554124668) q[4];
ry(-0.8431842000451076) q[5];
cx q[4],q[5];
ry(-0.4405841067652787) q[4];
ry(-2.478148965841494) q[5];
cx q[4],q[5];
ry(2.8744423610800665) q[5];
ry(-0.06876862210223145) q[6];
cx q[5],q[6];
ry(-0.643945951292614) q[5];
ry(2.6309580413097535) q[6];
cx q[5],q[6];
ry(-0.46349220552857595) q[6];
ry(1.0776718050766272) q[7];
cx q[6],q[7];
ry(-1.3418486311472007) q[6];
ry(-1.4597689644624978) q[7];
cx q[6],q[7];
ry(1.1555379188271333) q[0];
ry(-0.0013347881962231362) q[1];
cx q[0],q[1];
ry(1.5001652041224591) q[0];
ry(2.9453868582356972) q[1];
cx q[0],q[1];
ry(2.462220049821223) q[1];
ry(-1.6126838860542432) q[2];
cx q[1],q[2];
ry(-2.106245225728589) q[1];
ry(0.25062953221972606) q[2];
cx q[1],q[2];
ry(2.1807768325059316) q[2];
ry(-3.001589821415146) q[3];
cx q[2],q[3];
ry(-1.1409368888437816) q[2];
ry(-0.20706405768820227) q[3];
cx q[2],q[3];
ry(3.0756564621304245) q[3];
ry(1.4326989078182761) q[4];
cx q[3],q[4];
ry(-0.2885978343151488) q[3];
ry(-0.9572146969117531) q[4];
cx q[3],q[4];
ry(2.317205631225138) q[4];
ry(0.0713213747626007) q[5];
cx q[4],q[5];
ry(1.051230684803576) q[4];
ry(0.08666251353348427) q[5];
cx q[4],q[5];
ry(2.7001148018075396) q[5];
ry(1.7574240132552976) q[6];
cx q[5],q[6];
ry(-1.2095368717060309) q[5];
ry(-2.839322890247217) q[6];
cx q[5],q[6];
ry(1.2851747483611033) q[6];
ry(-1.987253616013561) q[7];
cx q[6],q[7];
ry(-2.699167655412433) q[6];
ry(-2.524207089155249) q[7];
cx q[6],q[7];
ry(2.1055681191731486) q[0];
ry(-2.6200523896185928) q[1];
cx q[0],q[1];
ry(2.2799009412876483) q[0];
ry(1.5284636326404462) q[1];
cx q[0],q[1];
ry(-1.204228648610697) q[1];
ry(0.4614108032535388) q[2];
cx q[1],q[2];
ry(-0.9517374830559154) q[1];
ry(2.2271453429880177) q[2];
cx q[1],q[2];
ry(-0.2728036147592823) q[2];
ry(-0.046689429999949184) q[3];
cx q[2],q[3];
ry(1.6227240086688681) q[2];
ry(-2.432680195002751) q[3];
cx q[2],q[3];
ry(2.1846272624305385) q[3];
ry(-2.891730796250002) q[4];
cx q[3],q[4];
ry(1.4803213225258178) q[3];
ry(-1.122093656295298) q[4];
cx q[3],q[4];
ry(-1.1085329423069346) q[4];
ry(-2.8705657110714267) q[5];
cx q[4],q[5];
ry(0.36831887186977946) q[4];
ry(-0.9956956440572446) q[5];
cx q[4],q[5];
ry(1.7987585539766786) q[5];
ry(-2.9689947343239247) q[6];
cx q[5],q[6];
ry(-0.3305068014373245) q[5];
ry(0.42664863862830593) q[6];
cx q[5],q[6];
ry(-2.6785056787436545) q[6];
ry(-1.4796034381904037) q[7];
cx q[6],q[7];
ry(-3.0577166514102823) q[6];
ry(-3.0152913778119785) q[7];
cx q[6],q[7];
ry(0.12281868123818927) q[0];
ry(-0.5921972250690607) q[1];
cx q[0],q[1];
ry(-1.5784352857247805) q[0];
ry(0.3516490773725412) q[1];
cx q[0],q[1];
ry(-0.08141993126574931) q[1];
ry(-2.0840837909642342) q[2];
cx q[1],q[2];
ry(-0.7292599861463002) q[1];
ry(3.022773689623698) q[2];
cx q[1],q[2];
ry(0.07754136127274157) q[2];
ry(2.7701188791798517) q[3];
cx q[2],q[3];
ry(2.3937964574616197) q[2];
ry(-0.9793783845820647) q[3];
cx q[2],q[3];
ry(-2.6148229299582755) q[3];
ry(-0.03947358120313519) q[4];
cx q[3],q[4];
ry(1.9679323788848215) q[3];
ry(0.7043266080229191) q[4];
cx q[3],q[4];
ry(1.9131306246603321) q[4];
ry(0.521095878046995) q[5];
cx q[4],q[5];
ry(-2.5777545905558616) q[4];
ry(-3.121269255225039) q[5];
cx q[4],q[5];
ry(-2.912151238867416) q[5];
ry(-0.7507416071615262) q[6];
cx q[5],q[6];
ry(2.315517494584328) q[5];
ry(2.7357295870383687) q[6];
cx q[5],q[6];
ry(-2.771982416053758) q[6];
ry(1.9238943258315524) q[7];
cx q[6],q[7];
ry(2.4554769428204115) q[6];
ry(2.1845200255249484) q[7];
cx q[6],q[7];
ry(-0.2577830722119282) q[0];
ry(-1.5379356628354637) q[1];
cx q[0],q[1];
ry(1.2640964995619381) q[0];
ry(3.135027171641613) q[1];
cx q[0],q[1];
ry(2.72550107619584) q[1];
ry(-1.180387347118435) q[2];
cx q[1],q[2];
ry(-1.5982868770502907) q[1];
ry(-1.7100272945340063) q[2];
cx q[1],q[2];
ry(2.828010578896705) q[2];
ry(-2.0800249333566274) q[3];
cx q[2],q[3];
ry(-0.9209037951286226) q[2];
ry(2.475093772362897) q[3];
cx q[2],q[3];
ry(2.812961292885783) q[3];
ry(-0.7756734972719581) q[4];
cx q[3],q[4];
ry(-3.007131352484367) q[3];
ry(-1.2676135722669732) q[4];
cx q[3],q[4];
ry(1.5334625029318776) q[4];
ry(-2.8224355010729028) q[5];
cx q[4],q[5];
ry(2.6901078152708426) q[4];
ry(-1.0351039333630527) q[5];
cx q[4],q[5];
ry(2.980672712021618) q[5];
ry(-0.4161389596221011) q[6];
cx q[5],q[6];
ry(-1.0476321905437302) q[5];
ry(-0.4541929422576363) q[6];
cx q[5],q[6];
ry(2.6043606263400876) q[6];
ry(0.6448015766129689) q[7];
cx q[6],q[7];
ry(1.3026278523620585) q[6];
ry(2.0405916264392814) q[7];
cx q[6],q[7];
ry(-1.282151810836785) q[0];
ry(0.17507717580569804) q[1];
cx q[0],q[1];
ry(2.4866584586243246) q[0];
ry(-0.8034193665350857) q[1];
cx q[0],q[1];
ry(0.7471718818627896) q[1];
ry(-2.04951867597964) q[2];
cx q[1],q[2];
ry(-2.4902777669963356) q[1];
ry(0.08968528032828971) q[2];
cx q[1],q[2];
ry(0.9202341169280961) q[2];
ry(1.354487882738237) q[3];
cx q[2],q[3];
ry(-1.6607843113329759) q[2];
ry(0.729130465040889) q[3];
cx q[2],q[3];
ry(3.0835062913087903) q[3];
ry(1.5089017489133236) q[4];
cx q[3],q[4];
ry(0.6908434215552949) q[3];
ry(-1.1178363708400905) q[4];
cx q[3],q[4];
ry(0.8965729794648376) q[4];
ry(0.28428590211764426) q[5];
cx q[4],q[5];
ry(0.14628661562353248) q[4];
ry(2.9997527787872778) q[5];
cx q[4],q[5];
ry(2.484970617456119) q[5];
ry(-1.8959281027545078) q[6];
cx q[5],q[6];
ry(2.255716996786938) q[5];
ry(1.4881769812411454) q[6];
cx q[5],q[6];
ry(0.547236073618067) q[6];
ry(0.6472008432852399) q[7];
cx q[6],q[7];
ry(-0.4297285189220812) q[6];
ry(-0.03653373234488989) q[7];
cx q[6],q[7];
ry(2.89863274816393) q[0];
ry(-1.2823068517307612) q[1];
cx q[0],q[1];
ry(2.8933686597447528) q[0];
ry(0.5226016131200897) q[1];
cx q[0],q[1];
ry(-2.737051455183916) q[1];
ry(-1.1041447401451903) q[2];
cx q[1],q[2];
ry(0.8737739888777822) q[1];
ry(1.3471157450472886) q[2];
cx q[1],q[2];
ry(-0.7112270789451918) q[2];
ry(0.0691386158836913) q[3];
cx q[2],q[3];
ry(2.7307472425261445) q[2];
ry(0.5691619685247584) q[3];
cx q[2],q[3];
ry(1.4790111747911336) q[3];
ry(-3.0684238290997303) q[4];
cx q[3],q[4];
ry(1.3082583693768297) q[3];
ry(-1.3750220575066179) q[4];
cx q[3],q[4];
ry(-1.979299133421259) q[4];
ry(-2.7410601704770503) q[5];
cx q[4],q[5];
ry(-1.2600386856484103) q[4];
ry(1.308485806047332) q[5];
cx q[4],q[5];
ry(-1.1384247909218068) q[5];
ry(-2.9828358660313286) q[6];
cx q[5],q[6];
ry(0.7133971668864649) q[5];
ry(2.1834406427287867) q[6];
cx q[5],q[6];
ry(0.4479866902448526) q[6];
ry(-2.869065968203792) q[7];
cx q[6],q[7];
ry(0.7760135818696137) q[6];
ry(2.215919711561519) q[7];
cx q[6],q[7];
ry(-0.9407810057553281) q[0];
ry(1.1339740418013253) q[1];
cx q[0],q[1];
ry(-2.626993505007224) q[0];
ry(1.6796426193605667) q[1];
cx q[0],q[1];
ry(-2.5799994117383442) q[1];
ry(2.900438602427231) q[2];
cx q[1],q[2];
ry(-1.254183091118001) q[1];
ry(-2.0459774136077287) q[2];
cx q[1],q[2];
ry(1.1448512600346674) q[2];
ry(2.2564708493896823) q[3];
cx q[2],q[3];
ry(1.3792825604361887) q[2];
ry(1.8711812066274591) q[3];
cx q[2],q[3];
ry(-0.031025487251532983) q[3];
ry(-1.9239746945517233) q[4];
cx q[3],q[4];
ry(-2.5552458619883156) q[3];
ry(-2.1970416101400447) q[4];
cx q[3],q[4];
ry(1.4561621645396858) q[4];
ry(-2.6425590238535444) q[5];
cx q[4],q[5];
ry(-1.273865926421516) q[4];
ry(2.3272035711846564) q[5];
cx q[4],q[5];
ry(-0.5037561820016335) q[5];
ry(-2.025267154504821) q[6];
cx q[5],q[6];
ry(-1.88699173483445) q[5];
ry(2.399026460824868) q[6];
cx q[5],q[6];
ry(2.0050939985894605) q[6];
ry(0.0875172504277364) q[7];
cx q[6],q[7];
ry(0.4383778329090707) q[6];
ry(1.783835990401772) q[7];
cx q[6],q[7];
ry(-2.1545157339360967) q[0];
ry(2.7850791974692064) q[1];
cx q[0],q[1];
ry(0.2943575669767175) q[0];
ry(1.6458840253197156) q[1];
cx q[0],q[1];
ry(-0.1738212109966728) q[1];
ry(-2.737064611670399) q[2];
cx q[1],q[2];
ry(0.9329449566913087) q[1];
ry(2.8278221364898872) q[2];
cx q[1],q[2];
ry(1.6687141319203374) q[2];
ry(-1.002894772709066) q[3];
cx q[2],q[3];
ry(0.11312592288186198) q[2];
ry(2.2221523350323666) q[3];
cx q[2],q[3];
ry(0.5129369702262033) q[3];
ry(0.9686568037962209) q[4];
cx q[3],q[4];
ry(2.966763126734204) q[3];
ry(-1.9188758748563717) q[4];
cx q[3],q[4];
ry(-0.7698572689905587) q[4];
ry(1.1319415991594732) q[5];
cx q[4],q[5];
ry(2.966088968556427) q[4];
ry(2.6848790862961365) q[5];
cx q[4],q[5];
ry(-0.40729744418183694) q[5];
ry(-0.38808109607958485) q[6];
cx q[5],q[6];
ry(2.6803390039811257) q[5];
ry(0.639452070756959) q[6];
cx q[5],q[6];
ry(0.332795775822412) q[6];
ry(-1.579011985725476) q[7];
cx q[6],q[7];
ry(0.012462442100011017) q[6];
ry(-2.4886539455823056) q[7];
cx q[6],q[7];
ry(-1.3547256191454515) q[0];
ry(0.16324162878396287) q[1];
cx q[0],q[1];
ry(-0.5223406860654016) q[0];
ry(-0.9537302935609056) q[1];
cx q[0],q[1];
ry(-2.9453984898483525) q[1];
ry(1.3369009016160724) q[2];
cx q[1],q[2];
ry(-2.4870830737353486) q[1];
ry(1.7251214042968686) q[2];
cx q[1],q[2];
ry(1.9768247920294675) q[2];
ry(0.3163896411466789) q[3];
cx q[2],q[3];
ry(1.8393752748047374) q[2];
ry(-1.1211984356228681) q[3];
cx q[2],q[3];
ry(-1.992313378650661) q[3];
ry(0.40594784200194844) q[4];
cx q[3],q[4];
ry(-0.048793087036271734) q[3];
ry(2.541372697905277) q[4];
cx q[3],q[4];
ry(0.16881454659798312) q[4];
ry(-1.3457091585117356) q[5];
cx q[4],q[5];
ry(1.0236901502418185) q[4];
ry(-2.6913936105876353) q[5];
cx q[4],q[5];
ry(1.0752779646438047) q[5];
ry(-2.4934659963998733) q[6];
cx q[5],q[6];
ry(-0.5126281079182274) q[5];
ry(1.2814321245560527) q[6];
cx q[5],q[6];
ry(-0.5063714865855813) q[6];
ry(1.2034538861176953) q[7];
cx q[6],q[7];
ry(-2.3666921603175957) q[6];
ry(1.3778535849488645) q[7];
cx q[6],q[7];
ry(-1.068389096078941) q[0];
ry(2.744588424612128) q[1];
cx q[0],q[1];
ry(1.0455094051819356) q[0];
ry(-2.2529378284813433) q[1];
cx q[0],q[1];
ry(-1.0828971578482811) q[1];
ry(-2.823071534207445) q[2];
cx q[1],q[2];
ry(1.2602261546088291) q[1];
ry(0.06314878349260233) q[2];
cx q[1],q[2];
ry(1.7781141218718828) q[2];
ry(-1.9427085505722523) q[3];
cx q[2],q[3];
ry(0.6233686652701131) q[2];
ry(1.3763308658407745) q[3];
cx q[2],q[3];
ry(0.44687552726143537) q[3];
ry(2.065160361125601) q[4];
cx q[3],q[4];
ry(1.203597480425998) q[3];
ry(0.49464643392860036) q[4];
cx q[3],q[4];
ry(-2.8235117044514) q[4];
ry(0.46396879944142) q[5];
cx q[4],q[5];
ry(-1.1889007261687379) q[4];
ry(-1.6706795004826758) q[5];
cx q[4],q[5];
ry(0.652911573839953) q[5];
ry(1.910359979769494) q[6];
cx q[5],q[6];
ry(-1.0549502523454757) q[5];
ry(-1.742508284752952) q[6];
cx q[5],q[6];
ry(-0.013692295342292115) q[6];
ry(1.3439792419167975) q[7];
cx q[6],q[7];
ry(-1.3563224362296538) q[6];
ry(1.8486275811675852) q[7];
cx q[6],q[7];
ry(2.4030305860726187) q[0];
ry(0.3228863333495964) q[1];
ry(2.496614698239571) q[2];
ry(-3.0364711120974497) q[3];
ry(1.3668313597567814) q[4];
ry(-2.454967936492652) q[5];
ry(-2.7341772881831288) q[6];
ry(2.1603427097825176) q[7];