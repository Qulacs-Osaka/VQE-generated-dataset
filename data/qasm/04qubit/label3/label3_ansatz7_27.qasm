OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.0229462379549585) q[0];
ry(-1.3234870674654147) q[1];
cx q[0],q[1];
ry(-1.0138531259710961) q[0];
ry(1.3649453555292155) q[1];
cx q[0],q[1];
ry(-2.099280157091731) q[0];
ry(0.9281272330015272) q[2];
cx q[0],q[2];
ry(1.0103730645605986) q[0];
ry(-0.8847886303636752) q[2];
cx q[0],q[2];
ry(-0.37165449949069096) q[0];
ry(-1.186944158151364) q[3];
cx q[0],q[3];
ry(2.0810664317865912) q[0];
ry(0.49118835417426515) q[3];
cx q[0],q[3];
ry(-1.8850299833617479) q[1];
ry(1.231656624861095) q[2];
cx q[1],q[2];
ry(2.392950413705063) q[1];
ry(2.09082578312415) q[2];
cx q[1],q[2];
ry(1.1467438912450958) q[1];
ry(-2.505848552073752) q[3];
cx q[1],q[3];
ry(-0.5030581938449208) q[1];
ry(-1.4344419624394797) q[3];
cx q[1],q[3];
ry(2.36031380236717) q[2];
ry(-2.0192105507553553) q[3];
cx q[2],q[3];
ry(-0.887909808714352) q[2];
ry(0.9509938095389003) q[3];
cx q[2],q[3];
ry(1.53501817773141) q[0];
ry(-1.2092394613047714) q[1];
cx q[0],q[1];
ry(1.8016158042694765) q[0];
ry(0.14389555299293025) q[1];
cx q[0],q[1];
ry(-0.389746579505652) q[0];
ry(3.012725328397401) q[2];
cx q[0],q[2];
ry(-1.5029143573786672) q[0];
ry(-2.8165439819366993) q[2];
cx q[0],q[2];
ry(-1.015630283761435) q[0];
ry(-1.8866624969033232) q[3];
cx q[0],q[3];
ry(-0.05782572542995082) q[0];
ry(0.578408142863245) q[3];
cx q[0],q[3];
ry(-0.23486658176417605) q[1];
ry(-1.5035040830536393) q[2];
cx q[1],q[2];
ry(-1.8393224195913171) q[1];
ry(1.5054003422464488) q[2];
cx q[1],q[2];
ry(1.9624313423783206) q[1];
ry(-2.4282047820600727) q[3];
cx q[1],q[3];
ry(-0.20489262244836812) q[1];
ry(1.8590864718722395) q[3];
cx q[1],q[3];
ry(1.4503829690615797) q[2];
ry(0.6689230548399957) q[3];
cx q[2],q[3];
ry(-0.5982397808092103) q[2];
ry(-1.8077642643066918) q[3];
cx q[2],q[3];
ry(-2.900670265901458) q[0];
ry(0.06873038395823095) q[1];
cx q[0],q[1];
ry(0.5739648820367828) q[0];
ry(-2.1214887408698058) q[1];
cx q[0],q[1];
ry(1.704799967109673) q[0];
ry(-2.062471862229713) q[2];
cx q[0],q[2];
ry(-1.4000188781040475) q[0];
ry(-1.6330659026394347) q[2];
cx q[0],q[2];
ry(-2.3610298391380837) q[0];
ry(-2.3728619747552924) q[3];
cx q[0],q[3];
ry(-1.8971104122207905) q[0];
ry(-1.6763160929847922) q[3];
cx q[0],q[3];
ry(-1.59666970458009) q[1];
ry(1.6235447195909511) q[2];
cx q[1],q[2];
ry(2.5096051532461696) q[1];
ry(0.9347519900558668) q[2];
cx q[1],q[2];
ry(-0.21033083857641002) q[1];
ry(2.180109312915717) q[3];
cx q[1],q[3];
ry(0.637358570941684) q[1];
ry(-1.536771652325304) q[3];
cx q[1],q[3];
ry(-2.561764624636562) q[2];
ry(2.7658572141310773) q[3];
cx q[2],q[3];
ry(-0.15875928239246304) q[2];
ry(2.6954716919917163) q[3];
cx q[2],q[3];
ry(-0.30080790975610494) q[0];
ry(1.562035677045814) q[1];
cx q[0],q[1];
ry(-2.184369234155823) q[0];
ry(-2.458599445199054) q[1];
cx q[0],q[1];
ry(-1.997741378789427) q[0];
ry(1.604352262451636) q[2];
cx q[0],q[2];
ry(1.0597000118037876) q[0];
ry(3.0971205021841244) q[2];
cx q[0],q[2];
ry(-2.4315114208911033) q[0];
ry(0.16121429782370406) q[3];
cx q[0],q[3];
ry(-1.6475136169893767) q[0];
ry(-2.369856943348099) q[3];
cx q[0],q[3];
ry(2.8524653193208267) q[1];
ry(1.501975507192177) q[2];
cx q[1],q[2];
ry(1.0493983875571016) q[1];
ry(2.847031800726452) q[2];
cx q[1],q[2];
ry(2.849126058277845) q[1];
ry(2.90072788451895) q[3];
cx q[1],q[3];
ry(0.4418861631785198) q[1];
ry(0.7553657910418762) q[3];
cx q[1],q[3];
ry(-2.6816150987707554) q[2];
ry(2.0797697089153226) q[3];
cx q[2],q[3];
ry(0.9852832087685508) q[2];
ry(-2.3142969008965526) q[3];
cx q[2],q[3];
ry(-2.069501933073081) q[0];
ry(-0.8322718642911724) q[1];
cx q[0],q[1];
ry(0.8780725664833081) q[0];
ry(-0.37004270176232357) q[1];
cx q[0],q[1];
ry(2.989390310511361) q[0];
ry(-2.0504577719184445) q[2];
cx q[0],q[2];
ry(-2.9920740077198427) q[0];
ry(-2.726524192998253) q[2];
cx q[0],q[2];
ry(-2.420151718282225) q[0];
ry(-1.5996448246829196) q[3];
cx q[0],q[3];
ry(-0.8890732654519873) q[0];
ry(1.2428605870672973) q[3];
cx q[0],q[3];
ry(-1.453033852529537) q[1];
ry(1.2293916407911318) q[2];
cx q[1],q[2];
ry(-1.124327910597593) q[1];
ry(-0.8759224217070476) q[2];
cx q[1],q[2];
ry(1.85274622969379) q[1];
ry(1.3358608305071655) q[3];
cx q[1],q[3];
ry(1.734785121594963) q[1];
ry(2.0147383908692817) q[3];
cx q[1],q[3];
ry(-1.7153651619367054) q[2];
ry(1.3141967951029359) q[3];
cx q[2],q[3];
ry(1.9084730217434585) q[2];
ry(0.747627731635956) q[3];
cx q[2],q[3];
ry(1.7272927267060387) q[0];
ry(-2.4172239446590034) q[1];
cx q[0],q[1];
ry(-1.1386200289505357) q[0];
ry(0.8061682866877629) q[1];
cx q[0],q[1];
ry(-1.295669658047479) q[0];
ry(1.6205654441398707) q[2];
cx q[0],q[2];
ry(2.622115470563593) q[0];
ry(2.8617855909731906) q[2];
cx q[0],q[2];
ry(1.5075160262504417) q[0];
ry(-0.6794274968490397) q[3];
cx q[0],q[3];
ry(0.5254442164549554) q[0];
ry(2.86293382573669) q[3];
cx q[0],q[3];
ry(-0.27105209088986054) q[1];
ry(-0.7475455935525198) q[2];
cx q[1],q[2];
ry(-2.514889696157297) q[1];
ry(-0.6067228249322616) q[2];
cx q[1],q[2];
ry(2.564280025835005) q[1];
ry(-2.7614087423493303) q[3];
cx q[1],q[3];
ry(-1.9718843274174729) q[1];
ry(-0.5844291134416025) q[3];
cx q[1],q[3];
ry(-0.6929062168143022) q[2];
ry(0.0625264868368385) q[3];
cx q[2],q[3];
ry(1.0844231464792395) q[2];
ry(-2.06770069380297) q[3];
cx q[2],q[3];
ry(-2.0472247645999886) q[0];
ry(1.0770716827882918) q[1];
cx q[0],q[1];
ry(3.1266175345204017) q[0];
ry(2.7146993113254787) q[1];
cx q[0],q[1];
ry(-3.028185050472179) q[0];
ry(-0.13318815171192888) q[2];
cx q[0],q[2];
ry(-0.3894512168452691) q[0];
ry(1.7257333048533638) q[2];
cx q[0],q[2];
ry(-2.687514490149199) q[0];
ry(-1.4252595909285013) q[3];
cx q[0],q[3];
ry(0.5893535858266706) q[0];
ry(-0.8403655933407648) q[3];
cx q[0],q[3];
ry(0.30912098200325866) q[1];
ry(1.7600783718727209) q[2];
cx q[1],q[2];
ry(-0.637899187650711) q[1];
ry(-2.8534850405841126) q[2];
cx q[1],q[2];
ry(-2.5339934641322897) q[1];
ry(1.1556721995952046) q[3];
cx q[1],q[3];
ry(-0.8773376725562274) q[1];
ry(1.069857931893191) q[3];
cx q[1],q[3];
ry(-0.9504844573933964) q[2];
ry(0.17373055905059023) q[3];
cx q[2],q[3];
ry(-1.6239168754432667) q[2];
ry(-0.4907566937600051) q[3];
cx q[2],q[3];
ry(-0.5595897228818689) q[0];
ry(1.5308481839928794) q[1];
cx q[0],q[1];
ry(-1.1818067375462658) q[0];
ry(-0.4888830242118738) q[1];
cx q[0],q[1];
ry(-0.461155083801461) q[0];
ry(-1.5292237593620681) q[2];
cx q[0],q[2];
ry(0.6567067201824841) q[0];
ry(-1.6225384687902125) q[2];
cx q[0],q[2];
ry(-2.7926230072094542) q[0];
ry(2.93117486561629) q[3];
cx q[0],q[3];
ry(1.9452967001243424) q[0];
ry(-0.5662014477153594) q[3];
cx q[0],q[3];
ry(-2.8809690267551797) q[1];
ry(1.8264713490165994) q[2];
cx q[1],q[2];
ry(0.2909037165507719) q[1];
ry(-0.9320862405850114) q[2];
cx q[1],q[2];
ry(-0.16381505435748195) q[1];
ry(-0.17695109170599554) q[3];
cx q[1],q[3];
ry(-1.826720831272909) q[1];
ry(-0.7351704333204008) q[3];
cx q[1],q[3];
ry(-2.7656996235442675) q[2];
ry(2.352162189312947) q[3];
cx q[2],q[3];
ry(-0.7764759483452695) q[2];
ry(1.935534975668193) q[3];
cx q[2],q[3];
ry(0.26248847639215867) q[0];
ry(-1.415816806730347) q[1];
cx q[0],q[1];
ry(-2.447225498464976) q[0];
ry(-2.7144554703999204) q[1];
cx q[0],q[1];
ry(2.1122616014438442) q[0];
ry(-3.0075225065858042) q[2];
cx q[0],q[2];
ry(0.7063860335918184) q[0];
ry(1.457204024852122) q[2];
cx q[0],q[2];
ry(-1.288381849009127) q[0];
ry(-1.9864413294937968) q[3];
cx q[0],q[3];
ry(-2.192636126547816) q[0];
ry(1.9750152041035705) q[3];
cx q[0],q[3];
ry(-1.9288046688016385) q[1];
ry(2.4196669103683437) q[2];
cx q[1],q[2];
ry(-1.0647001343266707) q[1];
ry(2.2687222616231937) q[2];
cx q[1],q[2];
ry(-2.8495249187853884) q[1];
ry(-0.6506964968677967) q[3];
cx q[1],q[3];
ry(-2.638433975063712) q[1];
ry(2.265276569799239) q[3];
cx q[1],q[3];
ry(1.4420970538206126) q[2];
ry(-0.8435778355599464) q[3];
cx q[2],q[3];
ry(1.373475688192566) q[2];
ry(0.043402334098479926) q[3];
cx q[2],q[3];
ry(2.0653696120861174) q[0];
ry(-1.4491841044993037) q[1];
cx q[0],q[1];
ry(1.0238350105321443) q[0];
ry(1.1590801004646438) q[1];
cx q[0],q[1];
ry(0.7675781929768402) q[0];
ry(3.067424685232442) q[2];
cx q[0],q[2];
ry(2.273519162229123) q[0];
ry(-0.6236673122311114) q[2];
cx q[0],q[2];
ry(1.0804333721580484) q[0];
ry(-1.1076614921938095) q[3];
cx q[0],q[3];
ry(1.7696959637665732) q[0];
ry(-0.03122977108399627) q[3];
cx q[0],q[3];
ry(0.11812885913039128) q[1];
ry(-2.8637893464338586) q[2];
cx q[1],q[2];
ry(3.130607870293866) q[1];
ry(2.3053467617909367) q[2];
cx q[1],q[2];
ry(-2.695585497367672) q[1];
ry(1.6123227661321515) q[3];
cx q[1],q[3];
ry(-1.6183665198451742) q[1];
ry(-1.4631896862274232) q[3];
cx q[1],q[3];
ry(-2.402457962292033) q[2];
ry(-0.8236190818903899) q[3];
cx q[2],q[3];
ry(-1.0536568734811125) q[2];
ry(-2.233529718675195) q[3];
cx q[2],q[3];
ry(-2.2764112800964753) q[0];
ry(1.0150400110343916) q[1];
cx q[0],q[1];
ry(2.0345364822874155) q[0];
ry(0.9344462912747248) q[1];
cx q[0],q[1];
ry(2.4316529816775847) q[0];
ry(1.1929826063326103) q[2];
cx q[0],q[2];
ry(-2.37450245420701) q[0];
ry(-1.5982802527835824) q[2];
cx q[0],q[2];
ry(1.0925991281625649) q[0];
ry(1.0516997210536747) q[3];
cx q[0],q[3];
ry(1.765594799534061) q[0];
ry(2.5312933923763414) q[3];
cx q[0],q[3];
ry(0.8455171508736707) q[1];
ry(2.8897043078201543) q[2];
cx q[1],q[2];
ry(-2.3887865557116257) q[1];
ry(-2.6492804017406404) q[2];
cx q[1],q[2];
ry(0.07212779391475088) q[1];
ry(0.19101809205707987) q[3];
cx q[1],q[3];
ry(1.352544900041144) q[1];
ry(0.04298787658984349) q[3];
cx q[1],q[3];
ry(-1.1468792591503978) q[2];
ry(-1.8467102426228867) q[3];
cx q[2],q[3];
ry(-2.7733200815964203) q[2];
ry(-2.818135721873948) q[3];
cx q[2],q[3];
ry(1.0356627298096708) q[0];
ry(2.5001976270666773) q[1];
cx q[0],q[1];
ry(0.7687128624318059) q[0];
ry(2.3349812149506795) q[1];
cx q[0],q[1];
ry(0.20778588970239387) q[0];
ry(2.388612250221241) q[2];
cx q[0],q[2];
ry(-1.1235618256095095) q[0];
ry(-0.7110941977032955) q[2];
cx q[0],q[2];
ry(-3.0610912488138844) q[0];
ry(-0.22301102315072097) q[3];
cx q[0],q[3];
ry(1.5498587699510975) q[0];
ry(2.789798585169014) q[3];
cx q[0],q[3];
ry(-2.817402917782179) q[1];
ry(1.608597270196627) q[2];
cx q[1],q[2];
ry(0.025803728527271197) q[1];
ry(0.5477550192999416) q[2];
cx q[1],q[2];
ry(0.7438274518317858) q[1];
ry(1.060049948261513) q[3];
cx q[1],q[3];
ry(-2.4262402226451147) q[1];
ry(-1.8361757359416107) q[3];
cx q[1],q[3];
ry(0.4843739192721346) q[2];
ry(0.6767189259139785) q[3];
cx q[2],q[3];
ry(2.6609042888063184) q[2];
ry(0.9837677200266484) q[3];
cx q[2],q[3];
ry(1.883990500575835) q[0];
ry(-2.499481807006612) q[1];
cx q[0],q[1];
ry(0.9878163051422169) q[0];
ry(-2.9583173731525165) q[1];
cx q[0],q[1];
ry(2.0181514748182527) q[0];
ry(0.7554833712943204) q[2];
cx q[0],q[2];
ry(-1.1368019173472366) q[0];
ry(2.633539895760173) q[2];
cx q[0],q[2];
ry(0.7392055148701142) q[0];
ry(-0.7011383864653471) q[3];
cx q[0],q[3];
ry(-2.983806037221536) q[0];
ry(1.7801034703862344) q[3];
cx q[0],q[3];
ry(-0.01600202865357403) q[1];
ry(2.9890258601193542) q[2];
cx q[1],q[2];
ry(-2.6104654073113704) q[1];
ry(1.5028088450660784) q[2];
cx q[1],q[2];
ry(-0.3636625295032463) q[1];
ry(1.6528868181200125) q[3];
cx q[1],q[3];
ry(-0.6071519439932123) q[1];
ry(2.8623932373549734) q[3];
cx q[1],q[3];
ry(-1.9056077599517725) q[2];
ry(-1.7959696348117937) q[3];
cx q[2],q[3];
ry(-1.6266019189505763) q[2];
ry(-2.732475775663462) q[3];
cx q[2],q[3];
ry(-1.7923578434506737) q[0];
ry(1.3467282886036362) q[1];
cx q[0],q[1];
ry(0.21865764370027402) q[0];
ry(-1.7346246972336559) q[1];
cx q[0],q[1];
ry(-2.7716226883814716) q[0];
ry(-2.889589804585565) q[2];
cx q[0],q[2];
ry(-2.1033717285375904) q[0];
ry(-1.4203599305785293) q[2];
cx q[0],q[2];
ry(-0.5666735620912939) q[0];
ry(-1.258684375782419) q[3];
cx q[0],q[3];
ry(0.6538013445156645) q[0];
ry(-1.396883383107666) q[3];
cx q[0],q[3];
ry(-1.137583806587677) q[1];
ry(0.6892880904887759) q[2];
cx q[1],q[2];
ry(1.5520089918088509) q[1];
ry(3.07729293609772) q[2];
cx q[1],q[2];
ry(0.9191142089459672) q[1];
ry(0.7060774521049931) q[3];
cx q[1],q[3];
ry(-0.14020380022446619) q[1];
ry(2.6868349219139036) q[3];
cx q[1],q[3];
ry(2.164932469870471) q[2];
ry(-1.5846665864171496) q[3];
cx q[2],q[3];
ry(1.1495404646040086) q[2];
ry(-2.030217507802967) q[3];
cx q[2],q[3];
ry(3.1277565299073222) q[0];
ry(-0.5479308247406394) q[1];
cx q[0],q[1];
ry(0.027998908474893724) q[0];
ry(0.45805203680781403) q[1];
cx q[0],q[1];
ry(-2.081825462880406) q[0];
ry(0.9450415610323505) q[2];
cx q[0],q[2];
ry(0.7671050034314533) q[0];
ry(1.2076161061409365) q[2];
cx q[0],q[2];
ry(-1.0857449851021517) q[0];
ry(2.1363167601869026) q[3];
cx q[0],q[3];
ry(0.1195711840812568) q[0];
ry(-1.4745816593073915) q[3];
cx q[0],q[3];
ry(1.4588125108220025) q[1];
ry(-0.25559700374492944) q[2];
cx q[1],q[2];
ry(-1.1929334859353877) q[1];
ry(0.46340043346669635) q[2];
cx q[1],q[2];
ry(-2.7649001764341232) q[1];
ry(-1.5559902696904762) q[3];
cx q[1],q[3];
ry(1.5972160348253936) q[1];
ry(-1.1034832632288842) q[3];
cx q[1],q[3];
ry(-1.897091860247313) q[2];
ry(-1.6358389753378544) q[3];
cx q[2],q[3];
ry(-2.202259206362955) q[2];
ry(3.080543774431896) q[3];
cx q[2],q[3];
ry(-2.3001308210077966) q[0];
ry(1.8210521959634598) q[1];
cx q[0],q[1];
ry(2.8800074579085115) q[0];
ry(-0.6070801130324881) q[1];
cx q[0],q[1];
ry(0.962767413513772) q[0];
ry(0.3823754213015693) q[2];
cx q[0],q[2];
ry(0.9472723412375162) q[0];
ry(-0.2812676946689306) q[2];
cx q[0],q[2];
ry(0.7870361663586491) q[0];
ry(0.08584302565668145) q[3];
cx q[0],q[3];
ry(1.7107842500207224) q[0];
ry(2.421472032015587) q[3];
cx q[0],q[3];
ry(2.63106084585353) q[1];
ry(1.6377038813002256) q[2];
cx q[1],q[2];
ry(-0.7224533804835129) q[1];
ry(0.6654066308197916) q[2];
cx q[1],q[2];
ry(0.17817260553245617) q[1];
ry(-2.822503121621688) q[3];
cx q[1],q[3];
ry(-2.461248669826644) q[1];
ry(-2.386327091566361) q[3];
cx q[1],q[3];
ry(-2.168447772452562) q[2];
ry(0.14543641939703747) q[3];
cx q[2],q[3];
ry(2.05933171304658) q[2];
ry(-2.3537190740762) q[3];
cx q[2],q[3];
ry(2.2086459375010294) q[0];
ry(-2.2043545785694723) q[1];
cx q[0],q[1];
ry(-2.3458922745439126) q[0];
ry(1.9936583749062091) q[1];
cx q[0],q[1];
ry(-2.885304069863345) q[0];
ry(2.4071522271680363) q[2];
cx q[0],q[2];
ry(2.7504192862384693) q[0];
ry(-1.489935781069959) q[2];
cx q[0],q[2];
ry(-3.0250585120729494) q[0];
ry(-0.3225509343897393) q[3];
cx q[0],q[3];
ry(-0.1735894231320145) q[0];
ry(-2.6059348751408113) q[3];
cx q[0],q[3];
ry(1.3764769393093008) q[1];
ry(1.719723568147306) q[2];
cx q[1],q[2];
ry(0.6976196764164042) q[1];
ry(2.0047762687033597) q[2];
cx q[1],q[2];
ry(-0.7262204751796952) q[1];
ry(0.5306228352149761) q[3];
cx q[1],q[3];
ry(0.39848403821426) q[1];
ry(-0.7113257245059765) q[3];
cx q[1],q[3];
ry(2.289519311937885) q[2];
ry(1.3270386916258932) q[3];
cx q[2],q[3];
ry(-1.6947325012738208) q[2];
ry(-2.1557100805615836) q[3];
cx q[2],q[3];
ry(2.142882523711859) q[0];
ry(2.0679198480588745) q[1];
cx q[0],q[1];
ry(0.2591981364648541) q[0];
ry(0.6688207165528102) q[1];
cx q[0],q[1];
ry(-2.8774809542139774) q[0];
ry(-1.996873617209209) q[2];
cx q[0],q[2];
ry(-2.194782158503228) q[0];
ry(-3.008416655312213) q[2];
cx q[0],q[2];
ry(-0.051761237220526723) q[0];
ry(-2.605885190632843) q[3];
cx q[0],q[3];
ry(2.061603981801647) q[0];
ry(1.9867654850053675) q[3];
cx q[0],q[3];
ry(1.0896303911917329) q[1];
ry(2.9275495846828323) q[2];
cx q[1],q[2];
ry(-2.239128267886069) q[1];
ry(-0.7980595227294006) q[2];
cx q[1],q[2];
ry(-0.4517768219691885) q[1];
ry(-2.3557044442754074) q[3];
cx q[1],q[3];
ry(-1.0199304388911345) q[1];
ry(-2.35647576275915) q[3];
cx q[1],q[3];
ry(-2.267027467333257) q[2];
ry(1.6896424760977589) q[3];
cx q[2],q[3];
ry(-0.9249500182269497) q[2];
ry(-2.38138236506151) q[3];
cx q[2],q[3];
ry(-0.5684174853423568) q[0];
ry(1.324088484785304) q[1];
cx q[0],q[1];
ry(-2.842192502814835) q[0];
ry(2.0938751816586616) q[1];
cx q[0],q[1];
ry(-1.710682225651949) q[0];
ry(1.2604605187156865) q[2];
cx q[0],q[2];
ry(-0.8256650154101023) q[0];
ry(0.35491923863066704) q[2];
cx q[0],q[2];
ry(-2.5866419094020374) q[0];
ry(-1.8446319273318128) q[3];
cx q[0],q[3];
ry(-0.239290349208522) q[0];
ry(-3.0734087647584682) q[3];
cx q[0],q[3];
ry(-1.610330579541861) q[1];
ry(-1.1136536652195668) q[2];
cx q[1],q[2];
ry(-2.8841490814711914) q[1];
ry(-1.3937379914035801) q[2];
cx q[1],q[2];
ry(-1.6873926596593223) q[1];
ry(-0.42960035381916617) q[3];
cx q[1],q[3];
ry(0.3625274856178704) q[1];
ry(2.1224196781429683) q[3];
cx q[1],q[3];
ry(0.5359655796361391) q[2];
ry(-0.7470374268255253) q[3];
cx q[2],q[3];
ry(1.6584137482844348) q[2];
ry(-2.894687402303221) q[3];
cx q[2],q[3];
ry(-2.354573500314549) q[0];
ry(-1.7961325810374626) q[1];
cx q[0],q[1];
ry(-2.6120186783845725) q[0];
ry(-2.040240187961814) q[1];
cx q[0],q[1];
ry(1.477102166857521) q[0];
ry(1.5959391328605879) q[2];
cx q[0],q[2];
ry(-1.1293324103885052) q[0];
ry(1.22891002360156) q[2];
cx q[0],q[2];
ry(0.8915108866167045) q[0];
ry(0.5954010374102547) q[3];
cx q[0],q[3];
ry(-0.14428595834061575) q[0];
ry(1.4768677386131381) q[3];
cx q[0],q[3];
ry(-3.0063450677270502) q[1];
ry(0.35288997095837343) q[2];
cx q[1],q[2];
ry(1.7996929027416442) q[1];
ry(2.309459449050163) q[2];
cx q[1],q[2];
ry(-2.9453468842494983) q[1];
ry(-0.03245224558462628) q[3];
cx q[1],q[3];
ry(3.012753615858673) q[1];
ry(0.5296903055936992) q[3];
cx q[1],q[3];
ry(-0.5514923823787976) q[2];
ry(2.2433376222664014) q[3];
cx q[2],q[3];
ry(2.6826424887846865) q[2];
ry(3.063156317516773) q[3];
cx q[2],q[3];
ry(-2.575772425113367) q[0];
ry(2.933800236031597) q[1];
cx q[0],q[1];
ry(1.5866882325271803) q[0];
ry(2.8026704917662184) q[1];
cx q[0],q[1];
ry(-1.0493226310284545) q[0];
ry(-2.29478160320694) q[2];
cx q[0],q[2];
ry(-2.902568716020452) q[0];
ry(1.972528666899926) q[2];
cx q[0],q[2];
ry(2.8517795622109947) q[0];
ry(1.626136989904829) q[3];
cx q[0],q[3];
ry(0.27327804715784915) q[0];
ry(-0.807628390211832) q[3];
cx q[0],q[3];
ry(-0.795035694975213) q[1];
ry(1.9765471877321417) q[2];
cx q[1],q[2];
ry(1.795921689924611) q[1];
ry(2.5586821429967195) q[2];
cx q[1],q[2];
ry(0.9400370268027629) q[1];
ry(2.8674308638071824) q[3];
cx q[1],q[3];
ry(-2.933261111511094) q[1];
ry(0.4642290182654598) q[3];
cx q[1],q[3];
ry(1.1946097221105134) q[2];
ry(-2.0407070777992917) q[3];
cx q[2],q[3];
ry(-0.7149499865729511) q[2];
ry(-1.9688557588662337) q[3];
cx q[2],q[3];
ry(2.703791924026486) q[0];
ry(1.5160388702161127) q[1];
cx q[0],q[1];
ry(-1.402447682991377) q[0];
ry(-2.7343376525767176) q[1];
cx q[0],q[1];
ry(2.8567275400178263) q[0];
ry(2.1769493218708753) q[2];
cx q[0],q[2];
ry(2.6315803876664425) q[0];
ry(0.9311457086117251) q[2];
cx q[0],q[2];
ry(0.17805077790267454) q[0];
ry(0.7983562729694916) q[3];
cx q[0],q[3];
ry(-0.21447205472231115) q[0];
ry(-2.135111487765453) q[3];
cx q[0],q[3];
ry(-0.8469453066305351) q[1];
ry(0.6025456215615383) q[2];
cx q[1],q[2];
ry(1.525482772923067) q[1];
ry(2.644110283323615) q[2];
cx q[1],q[2];
ry(0.3684728488777862) q[1];
ry(2.3714579500634216) q[3];
cx q[1],q[3];
ry(2.9141353431530175) q[1];
ry(-2.8166877840348974) q[3];
cx q[1],q[3];
ry(2.404276427079912) q[2];
ry(-1.3419575098564875) q[3];
cx q[2],q[3];
ry(2.838501042835537) q[2];
ry(0.07350355899840781) q[3];
cx q[2],q[3];
ry(-0.9541236429129368) q[0];
ry(2.032973591999645) q[1];
cx q[0],q[1];
ry(-2.9434075807829743) q[0];
ry(0.6582571155931651) q[1];
cx q[0],q[1];
ry(0.7880985250259442) q[0];
ry(0.18580332011282327) q[2];
cx q[0],q[2];
ry(1.169873540913911) q[0];
ry(-2.033143402773336) q[2];
cx q[0],q[2];
ry(0.6178009678871552) q[0];
ry(-2.2310311773828397) q[3];
cx q[0],q[3];
ry(0.965755604556306) q[0];
ry(-1.9266482378079246) q[3];
cx q[0],q[3];
ry(2.477458737114271) q[1];
ry(-2.6380018834715147) q[2];
cx q[1],q[2];
ry(-1.9339971994943557) q[1];
ry(-0.31022094797079536) q[2];
cx q[1],q[2];
ry(-3.04730102859618) q[1];
ry(1.0731253873376625) q[3];
cx q[1],q[3];
ry(0.8594115072280708) q[1];
ry(0.44183449343445336) q[3];
cx q[1],q[3];
ry(2.655416596152477) q[2];
ry(-1.417919481528155) q[3];
cx q[2],q[3];
ry(-2.7144874781426105) q[2];
ry(1.4533096225413917) q[3];
cx q[2],q[3];
ry(0.8943734301829568) q[0];
ry(-1.709598415033135) q[1];
cx q[0],q[1];
ry(-1.5283069330744385) q[0];
ry(2.771908158971621) q[1];
cx q[0],q[1];
ry(-1.2054270125306223) q[0];
ry(-0.7059751639721229) q[2];
cx q[0],q[2];
ry(1.1180138534313642) q[0];
ry(0.8547107370268038) q[2];
cx q[0],q[2];
ry(0.9060972944566607) q[0];
ry(-0.7565410697067284) q[3];
cx q[0],q[3];
ry(-1.810633725085193) q[0];
ry(-1.1875532183982536) q[3];
cx q[0],q[3];
ry(1.4291167747278448) q[1];
ry(-1.615257040571259) q[2];
cx q[1],q[2];
ry(0.20913789927612345) q[1];
ry(2.49115285535168) q[2];
cx q[1],q[2];
ry(-2.18234690675777) q[1];
ry(2.6755010741689778) q[3];
cx q[1],q[3];
ry(-3.0593802480701338) q[1];
ry(2.91056439434006) q[3];
cx q[1],q[3];
ry(0.13396253312097886) q[2];
ry(0.7782863239620763) q[3];
cx q[2],q[3];
ry(0.4278061579777157) q[2];
ry(-0.980318896842169) q[3];
cx q[2],q[3];
ry(2.026255373100172) q[0];
ry(-0.9526271001238185) q[1];
cx q[0],q[1];
ry(-0.5310577550851869) q[0];
ry(0.8023145389631052) q[1];
cx q[0],q[1];
ry(1.1639532379620805) q[0];
ry(0.8423885408949027) q[2];
cx q[0],q[2];
ry(2.7231753156255127) q[0];
ry(1.913013212479238) q[2];
cx q[0],q[2];
ry(0.0549268033095459) q[0];
ry(2.235144098328277) q[3];
cx q[0],q[3];
ry(0.538311418062648) q[0];
ry(-2.03821913962239) q[3];
cx q[0],q[3];
ry(2.9511552223017645) q[1];
ry(0.8265478229324491) q[2];
cx q[1],q[2];
ry(-2.678356111262043) q[1];
ry(2.9720176294992595) q[2];
cx q[1],q[2];
ry(2.4643995267165986) q[1];
ry(0.5873324539604599) q[3];
cx q[1],q[3];
ry(0.9398404593771348) q[1];
ry(-1.6469578405512977) q[3];
cx q[1],q[3];
ry(1.4636018421081145) q[2];
ry(0.4026734274494731) q[3];
cx q[2],q[3];
ry(2.6646713155611277) q[2];
ry(-3.0825736243152235) q[3];
cx q[2],q[3];
ry(-1.248422655134953) q[0];
ry(-1.8359627148372466) q[1];
cx q[0],q[1];
ry(-0.25817633627757147) q[0];
ry(2.175527134428962) q[1];
cx q[0],q[1];
ry(2.5559106518448957) q[0];
ry(-2.394370986964107) q[2];
cx q[0],q[2];
ry(-1.9137956050461327) q[0];
ry(-0.2892165694932546) q[2];
cx q[0],q[2];
ry(-1.199909060337096) q[0];
ry(2.2417821430123173) q[3];
cx q[0],q[3];
ry(-0.25140485568743576) q[0];
ry(2.067510879340669) q[3];
cx q[0],q[3];
ry(1.1561961955665538) q[1];
ry(-1.4161902656541496) q[2];
cx q[1],q[2];
ry(2.2038214506667577) q[1];
ry(-2.400833558796413) q[2];
cx q[1],q[2];
ry(-1.526378409832738) q[1];
ry(2.9354016133130356) q[3];
cx q[1],q[3];
ry(-1.3592709090269823) q[1];
ry(1.3509903586744088) q[3];
cx q[1],q[3];
ry(2.656568780947581) q[2];
ry(-2.2013559084612773) q[3];
cx q[2],q[3];
ry(0.31404887171709994) q[2];
ry(-2.0574206022050348) q[3];
cx q[2],q[3];
ry(-1.4157519012882966) q[0];
ry(0.3717773955848047) q[1];
cx q[0],q[1];
ry(2.8602660833111893) q[0];
ry(-0.769961139078621) q[1];
cx q[0],q[1];
ry(-2.0191850282131085) q[0];
ry(-1.5039482840334673) q[2];
cx q[0],q[2];
ry(2.3358209929984532) q[0];
ry(-2.60530992851219) q[2];
cx q[0],q[2];
ry(1.6778191253470087) q[0];
ry(0.4541039679949401) q[3];
cx q[0],q[3];
ry(2.499725840339996) q[0];
ry(-2.132529708802516) q[3];
cx q[0],q[3];
ry(-1.3017128163411904) q[1];
ry(-0.5882402932179032) q[2];
cx q[1],q[2];
ry(-2.961104417936766) q[1];
ry(-2.3838246886706065) q[2];
cx q[1],q[2];
ry(-3.0451963933191752) q[1];
ry(-0.16359000205487198) q[3];
cx q[1],q[3];
ry(-0.773580821314343) q[1];
ry(0.12000304297343245) q[3];
cx q[1],q[3];
ry(-2.673086231166935) q[2];
ry(-0.06044944497334431) q[3];
cx q[2],q[3];
ry(-2.6767763548968624) q[2];
ry(-1.330266479500297) q[3];
cx q[2],q[3];
ry(1.2025876304641159) q[0];
ry(-1.709288565967177) q[1];
cx q[0],q[1];
ry(2.295415970128838) q[0];
ry(-2.082788293129533) q[1];
cx q[0],q[1];
ry(2.86347221395418) q[0];
ry(2.873309625656712) q[2];
cx q[0],q[2];
ry(2.9384958285359355) q[0];
ry(-2.822374387353653) q[2];
cx q[0],q[2];
ry(1.0035454229023006) q[0];
ry(-2.0828553855697516) q[3];
cx q[0],q[3];
ry(2.8721946734100836) q[0];
ry(-1.8559073863804534) q[3];
cx q[0],q[3];
ry(2.099551678517032) q[1];
ry(2.9535182863122285) q[2];
cx q[1],q[2];
ry(0.8886694932062325) q[1];
ry(2.069963464473692) q[2];
cx q[1],q[2];
ry(-1.7093697153646519) q[1];
ry(1.963865026431705) q[3];
cx q[1],q[3];
ry(1.3036527066466979) q[1];
ry(1.4048955674114598) q[3];
cx q[1],q[3];
ry(-1.9548934971440666) q[2];
ry(-1.0657759450912412) q[3];
cx q[2],q[3];
ry(2.3515271623989333) q[2];
ry(-0.8618627864347065) q[3];
cx q[2],q[3];
ry(0.821339367875616) q[0];
ry(-2.9151584000909936) q[1];
cx q[0],q[1];
ry(2.6964802941163293) q[0];
ry(-2.6185497030506415) q[1];
cx q[0],q[1];
ry(1.0540620174589446) q[0];
ry(2.4344736525919353) q[2];
cx q[0],q[2];
ry(2.6028153678363752) q[0];
ry(-1.5968654386799008) q[2];
cx q[0],q[2];
ry(-2.651434405762794) q[0];
ry(-2.6494024197792787) q[3];
cx q[0],q[3];
ry(2.0926834010289674) q[0];
ry(-1.4773406102913413) q[3];
cx q[0],q[3];
ry(-0.11096543478275933) q[1];
ry(0.4992270227963793) q[2];
cx q[1],q[2];
ry(-3.0909857336448576) q[1];
ry(1.8394612655422249) q[2];
cx q[1],q[2];
ry(1.0381402391675554) q[1];
ry(2.322572761283932) q[3];
cx q[1],q[3];
ry(2.444461479537119) q[1];
ry(0.5860623305689643) q[3];
cx q[1],q[3];
ry(-3.026029786313654) q[2];
ry(-1.9511446931920053) q[3];
cx q[2],q[3];
ry(-0.4247840513365535) q[2];
ry(0.5615601613854153) q[3];
cx q[2],q[3];
ry(0.5914471973984774) q[0];
ry(-1.1418806157238497) q[1];
cx q[0],q[1];
ry(-2.87099655566343) q[0];
ry(-1.2412766549848335) q[1];
cx q[0],q[1];
ry(-2.8519918520440077) q[0];
ry(-1.616806576218027) q[2];
cx q[0],q[2];
ry(-2.791767272870701) q[0];
ry(-2.261881149390293) q[2];
cx q[0],q[2];
ry(-0.9669452398988322) q[0];
ry(1.843723796385683) q[3];
cx q[0],q[3];
ry(1.0190749963177819) q[0];
ry(0.48296867634613366) q[3];
cx q[0],q[3];
ry(-2.528259989497913) q[1];
ry(1.3127319578846999) q[2];
cx q[1],q[2];
ry(-3.039210423571242) q[1];
ry(2.829176010634371) q[2];
cx q[1],q[2];
ry(-1.805953801517492) q[1];
ry(2.9424629116446477) q[3];
cx q[1],q[3];
ry(-0.8411007650408715) q[1];
ry(-1.360673337925281) q[3];
cx q[1],q[3];
ry(-2.390878529455624) q[2];
ry(-0.3726800834291666) q[3];
cx q[2],q[3];
ry(1.9581930625282045) q[2];
ry(-0.48265984930074785) q[3];
cx q[2],q[3];
ry(0.695831684583208) q[0];
ry(-0.48774702629755556) q[1];
ry(0.055165902329852586) q[2];
ry(-0.7262575269954432) q[3];