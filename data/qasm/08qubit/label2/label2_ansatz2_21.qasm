OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.449721410080662) q[0];
rz(1.2414293433379404) q[0];
ry(-2.3923512759210204) q[1];
rz(-2.482750365536773) q[1];
ry(0.25798652240451214) q[2];
rz(-0.8143336109066265) q[2];
ry(0.9226653822061058) q[3];
rz(1.0745762293004875) q[3];
ry(1.834948742645186) q[4];
rz(-2.385779223812951) q[4];
ry(0.4760425722325171) q[5];
rz(-2.069932957836239) q[5];
ry(1.1065729302636131) q[6];
rz(0.29168434103013213) q[6];
ry(-1.08424446809761) q[7];
rz(0.9242239932848165) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.9921685946745291) q[0];
rz(1.9721824523006684) q[0];
ry(-0.24098038220543747) q[1];
rz(-1.3236543892824493) q[1];
ry(-2.3770188045534137) q[2];
rz(0.4347027268836889) q[2];
ry(2.4662248422126747) q[3];
rz(-1.2165971292937867) q[3];
ry(1.1679496316248414) q[4];
rz(1.5475289312519591) q[4];
ry(1.1947587169722338) q[5];
rz(2.223045873334141) q[5];
ry(-0.2509374305242176) q[6];
rz(-0.1323017756176892) q[6];
ry(-2.5718518496991236) q[7];
rz(1.1707628600205435) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.20057761293527587) q[0];
rz(2.1009239872862935) q[0];
ry(0.6048379419326322) q[1];
rz(-2.028497198212981) q[1];
ry(-1.1962867443542766) q[2];
rz(-0.5692388414742959) q[2];
ry(0.5952100352444898) q[3];
rz(-0.49464422658100554) q[3];
ry(-0.6067842296654224) q[4];
rz(1.3813938167818378) q[4];
ry(-2.822705040687147) q[5];
rz(-2.7212180751072705) q[5];
ry(2.177388892173924) q[6];
rz(-0.1757995369079719) q[6];
ry(-0.8173513235996587) q[7];
rz(-1.2267269422826992) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.971567202946339) q[0];
rz(-2.7785254833156263) q[0];
ry(1.8494053864791802) q[1];
rz(-0.13617883080764173) q[1];
ry(1.1968403902257365) q[2];
rz(-0.2143345215458021) q[2];
ry(2.5855703603677487) q[3];
rz(-2.1448317466757274) q[3];
ry(2.7765253786322686) q[4];
rz(-3.06930342997572) q[4];
ry(-0.4517393702301753) q[5];
rz(2.246763942563053) q[5];
ry(0.7163834457811279) q[6];
rz(3.1217109878561766) q[6];
ry(-2.630080183150011) q[7];
rz(-2.8333995993897254) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.7546066649578448) q[0];
rz(-2.4037572757506322) q[0];
ry(-1.519265802621723) q[1];
rz(0.7269883118516893) q[1];
ry(0.6298135883621283) q[2];
rz(0.10597931058132781) q[2];
ry(-1.7749949388899182) q[3];
rz(2.862860085614396) q[3];
ry(0.7186774515415342) q[4];
rz(-1.0466951856013376) q[4];
ry(-1.1788022618974174) q[5];
rz(-1.9326104681120737) q[5];
ry(2.350121803018072) q[6];
rz(0.470461509970983) q[6];
ry(-1.0301119547980582) q[7];
rz(-0.8774631671451846) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.817213298251805) q[0];
rz(-1.222688146845516) q[0];
ry(-2.1854287932501046) q[1];
rz(1.2922228047207274) q[1];
ry(1.5861322191474843) q[2];
rz(-1.907788309413829) q[2];
ry(-0.686124821565159) q[3];
rz(2.331974129542605) q[3];
ry(2.529427027625817) q[4];
rz(1.7098869026411412) q[4];
ry(-2.088529061825276) q[5];
rz(-2.8231802534793284) q[5];
ry(-1.425488871778544) q[6];
rz(-1.0074118125338343) q[6];
ry(-2.7790716044576946) q[7];
rz(0.9554666661405785) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.783015890057711) q[0];
rz(2.3226199474676155) q[0];
ry(-1.3016703180238673) q[1];
rz(-1.587936111267604) q[1];
ry(0.5697745873372397) q[2];
rz(1.7715073442204956) q[2];
ry(-1.2434330627971688) q[3];
rz(1.9308521483762906) q[3];
ry(2.1483149702606994) q[4];
rz(-2.7581007696402464) q[4];
ry(1.193121320362783) q[5];
rz(-0.06643351458393248) q[5];
ry(3.080441802447548) q[6];
rz(-2.694773268902999) q[6];
ry(2.6950376003773404) q[7];
rz(-0.5280169838092652) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.644990712225108) q[0];
rz(0.11032715855374596) q[0];
ry(0.30085246481109795) q[1];
rz(-2.824085401227587) q[1];
ry(2.0992023961578656) q[2];
rz(-2.148379339240301) q[2];
ry(2.499173555955059) q[3];
rz(-0.07846307586597807) q[3];
ry(1.8110141298315592) q[4];
rz(0.22092302400414798) q[4];
ry(-1.6610459159494282) q[5];
rz(0.21519399475172032) q[5];
ry(2.7607582246787814) q[6];
rz(2.8165727750268013) q[6];
ry(-0.062176218223713775) q[7];
rz(-1.1057103793333463) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.3685842194861297) q[0];
rz(-2.276519415061597) q[0];
ry(-0.9612439649362443) q[1];
rz(1.9279955484235174) q[1];
ry(-0.8407151607234414) q[2];
rz(0.2590355607157665) q[2];
ry(0.5307466125513365) q[3];
rz(3.1201432221353413) q[3];
ry(-0.9631403576804699) q[4];
rz(-1.1323588032675007) q[4];
ry(2.9741935137524136) q[5];
rz(1.9365338891210593) q[5];
ry(-0.7134351539106553) q[6];
rz(-0.12092879893144448) q[6];
ry(-0.9030539165388038) q[7];
rz(0.06895338992778655) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.1891422578815027) q[0];
rz(-2.199485672283105) q[0];
ry(-2.359053240469966) q[1];
rz(-0.31862986493309703) q[1];
ry(1.7499385567283974) q[2];
rz(-2.7183880137787257) q[2];
ry(1.4608768911272865) q[3];
rz(1.6219204285113342) q[3];
ry(2.5818401236852693) q[4];
rz(-0.3694640991179021) q[4];
ry(1.1299674515949067) q[5];
rz(-2.7116983487002706) q[5];
ry(-2.1436045692561496) q[6];
rz(-2.748525612144979) q[6];
ry(2.9076798365958045) q[7];
rz(2.9376958168032243) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.782701968907353) q[0];
rz(1.1132082910543877) q[0];
ry(-1.9743821822497445) q[1];
rz(2.589120121319775) q[1];
ry(1.864108281665666) q[2];
rz(-1.2680488753033794) q[2];
ry(-1.5559287503654406) q[3];
rz(-2.5139498681399663) q[3];
ry(-2.117052595275117) q[4];
rz(3.018649792074938) q[4];
ry(-2.39730944863156) q[5];
rz(-0.1756734311224848) q[5];
ry(-0.5394902266491023) q[6];
rz(-2.122944428775407) q[6];
ry(-0.2796538303066711) q[7];
rz(0.5865848157692486) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.0919545760531475) q[0];
rz(-2.664878638503978) q[0];
ry(-1.7006502403925816) q[1];
rz(0.10863064299963943) q[1];
ry(-2.8966720084984745) q[2];
rz(0.5279565377104118) q[2];
ry(2.1109069793647226) q[3];
rz(0.5116985589912041) q[3];
ry(-0.8673107443642429) q[4];
rz(-2.984647953949022) q[4];
ry(0.23865408378402025) q[5];
rz(2.804730111286242) q[5];
ry(-1.8195401489697662) q[6];
rz(-1.3235035205743557) q[6];
ry(1.1675255643482592) q[7];
rz(-1.869172617791587) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.2868699087103318) q[0];
rz(0.3106192789497877) q[0];
ry(2.336931117232515) q[1];
rz(-1.834472759649637) q[1];
ry(-2.7934472900803358) q[2];
rz(1.395392948193793) q[2];
ry(-3.0778323253637683) q[3];
rz(-1.911987778438679) q[3];
ry(-0.8453494425221777) q[4];
rz(1.3036457636781185) q[4];
ry(1.445941553736077) q[5];
rz(0.9308828488731908) q[5];
ry(-2.0001027416738753) q[6];
rz(-2.8463550266733697) q[6];
ry(-2.70702313149299) q[7];
rz(-2.9184789789050303) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.5155782628967276) q[0];
rz(1.9139775525166032) q[0];
ry(-1.0718795363496705) q[1];
rz(2.3812492270945955) q[1];
ry(-2.131557042833487) q[2];
rz(1.18937155736002) q[2];
ry(0.9757972898265758) q[3];
rz(-1.3596006738875128) q[3];
ry(2.6057402030844248) q[4];
rz(-1.6169191410204906) q[4];
ry(-2.3913532766295704) q[5];
rz(-0.7457067257906851) q[5];
ry(0.6896149220235562) q[6];
rz(0.8556571226821853) q[6];
ry(-0.955222342798869) q[7];
rz(0.6893064412290278) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.14319576496913378) q[0];
rz(2.1283062721437673) q[0];
ry(2.6787273553150603) q[1];
rz(-2.645754174297158) q[1];
ry(1.1801440139112354) q[2];
rz(0.5000618921658011) q[2];
ry(-1.1269828747489101) q[3];
rz(1.930852072779919) q[3];
ry(-3.0609847343771217) q[4];
rz(-2.8407152589990785) q[4];
ry(1.7798111848855374) q[5];
rz(1.6600469540470275) q[5];
ry(1.689065311585477) q[6];
rz(2.624750978694183) q[6];
ry(1.5302322657454508) q[7];
rz(-2.3036058451569916) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.8644923269473956) q[0];
rz(-0.37731009143486016) q[0];
ry(0.540052684959563) q[1];
rz(2.8646331020121236) q[1];
ry(1.479029045079673) q[2];
rz(-1.932498563091227) q[2];
ry(1.9893789588569746) q[3];
rz(-1.4350030543494228) q[3];
ry(-0.24979284077876152) q[4];
rz(-0.5507928792269954) q[4];
ry(0.7218058690875181) q[5];
rz(-0.3873819798991551) q[5];
ry(-3.033177442062168) q[6];
rz(-3.1250964568056196) q[6];
ry(-3.012864577284477) q[7];
rz(1.4156044905352778) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.30138490886942204) q[0];
rz(-2.699458625875317) q[0];
ry(2.237789363181727) q[1];
rz(1.1405366908411425) q[1];
ry(2.628814521979242) q[2];
rz(-3.1258271454250637) q[2];
ry(-0.9789161375898328) q[3];
rz(-0.19555207714628867) q[3];
ry(1.969249349322066) q[4];
rz(0.5028421908626752) q[4];
ry(2.2808129339287193) q[5];
rz(-2.3791714574433716) q[5];
ry(-1.247163111134605) q[6];
rz(-1.2149731443582334) q[6];
ry(-1.305306230090233) q[7];
rz(-2.572477480759387) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.05834893724194857) q[0];
rz(2.39534511399395) q[0];
ry(-1.162365494289502) q[1];
rz(-1.3057088952070677) q[1];
ry(-1.3244722423855821) q[2];
rz(1.475025941589753) q[2];
ry(-2.167092357732236) q[3];
rz(2.363747269897408) q[3];
ry(2.3666697096492597) q[4];
rz(0.959973622338289) q[4];
ry(-1.722406591731167) q[5];
rz(1.6836902462974406) q[5];
ry(-2.007031672920955) q[6];
rz(1.2417109838905365) q[6];
ry(0.8386753759189318) q[7];
rz(-2.248041638911424) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.407479504096196) q[0];
rz(-1.3493008333795888) q[0];
ry(-0.9883409303916599) q[1];
rz(1.6301233294032196) q[1];
ry(-2.430065459897853) q[2];
rz(-2.4754014440277716) q[2];
ry(-0.6533483255631065) q[3];
rz(-1.2896962307403148) q[3];
ry(2.105865848933712) q[4];
rz(-2.521746029108526) q[4];
ry(-2.7756147405990332) q[5];
rz(0.7639025805073151) q[5];
ry(3.022002093674087) q[6];
rz(-2.917464341509333) q[6];
ry(-0.7033966390554557) q[7];
rz(2.348027495585115) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.3290266546835348) q[0];
rz(1.0131915953522153) q[0];
ry(-1.6043756886397793) q[1];
rz(-1.460363832474415) q[1];
ry(2.380020022225044) q[2];
rz(0.868294589955218) q[2];
ry(-2.2818390912994566) q[3];
rz(1.619357043772121) q[3];
ry(-0.2712009664485878) q[4];
rz(-2.645763361915638) q[4];
ry(1.7397368980562105) q[5];
rz(2.5118979261406027) q[5];
ry(2.7702215539495336) q[6];
rz(0.16468332374751868) q[6];
ry(-1.4595281821064239) q[7];
rz(0.6428242608925019) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.7225605862542962) q[0];
rz(0.3916504030282979) q[0];
ry(0.11545637656158192) q[1];
rz(2.2690366779358797) q[1];
ry(-1.2935461762282008) q[2];
rz(1.3996165987373113) q[2];
ry(-2.0667035269150915) q[3];
rz(0.47069474846430276) q[3];
ry(-1.2936968570721215) q[4];
rz(-2.2455142128123393) q[4];
ry(-2.4947564066928662) q[5];
rz(-2.6613185271771753) q[5];
ry(-2.3072883897639658) q[6];
rz(-0.30251577901320503) q[6];
ry(1.4570920867752033) q[7];
rz(-1.4462353293889834) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.4648432484344789) q[0];
rz(-1.7155100070508533) q[0];
ry(1.0393749671323342) q[1];
rz(-1.128050895668344) q[1];
ry(0.22685058418415122) q[2];
rz(-0.9646070882614496) q[2];
ry(-2.2209584133392335) q[3];
rz(-0.2472633876555795) q[3];
ry(2.6792273529973754) q[4];
rz(-2.8137935071602027) q[4];
ry(-0.16959311369415708) q[5];
rz(-1.4500556715051518) q[5];
ry(-0.9154849072795392) q[6];
rz(-1.4695808269127761) q[6];
ry(-0.2065795151347354) q[7];
rz(-1.6209941585989656) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.6454767700291864) q[0];
rz(2.1055462919884027) q[0];
ry(1.193702156750624) q[1];
rz(1.1520511315746056) q[1];
ry(-1.798689853989794) q[2];
rz(-0.008577836402568373) q[2];
ry(-2.954302391044351) q[3];
rz(2.0241598652391435) q[3];
ry(-0.3547217529021012) q[4];
rz(-0.5479737498687953) q[4];
ry(0.6105033796543309) q[5];
rz(1.8002704830031888) q[5];
ry(2.374796187458111) q[6];
rz(-0.013749220785942205) q[6];
ry(-2.1050190994128437) q[7];
rz(-0.5327035716420845) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.0989452827704769) q[0];
rz(0.6333161044390482) q[0];
ry(-1.430866828997859) q[1];
rz(-1.4501789486022785) q[1];
ry(-2.022518600657161) q[2];
rz(-0.5978589980391318) q[2];
ry(-0.3210292262517731) q[3];
rz(-0.30563390729123) q[3];
ry(-3.04140435807463) q[4];
rz(-1.4078129022473427) q[4];
ry(2.0729485263773637) q[5];
rz(-2.5963852761510653) q[5];
ry(-1.4186729891916035) q[6];
rz(2.1098327353604116) q[6];
ry(2.5819870321161145) q[7];
rz(-1.1671772836995062) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.5057692208580171) q[0];
rz(2.8507601744924074) q[0];
ry(-1.9257989005032794) q[1];
rz(-0.8661523437133916) q[1];
ry(2.4469975330153426) q[2];
rz(2.5583821421286337) q[2];
ry(2.2325618145663206) q[3];
rz(0.35510028939208227) q[3];
ry(0.414749224917788) q[4];
rz(-1.3519069682516418) q[4];
ry(-0.33136362251036205) q[5];
rz(-2.970625777968414) q[5];
ry(3.085011286388687) q[6];
rz(1.3929889322114508) q[6];
ry(-2.4261931665121645) q[7];
rz(2.187409081070552) q[7];