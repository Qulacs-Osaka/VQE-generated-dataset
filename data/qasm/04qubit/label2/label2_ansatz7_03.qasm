OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.386770346110836) q[0];
ry(2.736903259843595) q[1];
cx q[0],q[1];
ry(2.576751330241607) q[0];
ry(-2.5428815130801126) q[1];
cx q[0],q[1];
ry(2.82508975231229) q[0];
ry(-1.591297385769309) q[2];
cx q[0],q[2];
ry(-1.3754346616223068) q[0];
ry(0.6275497108166244) q[2];
cx q[0],q[2];
ry(-1.4382496415576806) q[0];
ry(0.06363556277685412) q[3];
cx q[0],q[3];
ry(1.0880644048929693) q[0];
ry(-1.5885997412795962) q[3];
cx q[0],q[3];
ry(2.8191423272926337) q[1];
ry(1.5863516912026505) q[2];
cx q[1],q[2];
ry(3.0280344304769597) q[1];
ry(-2.663803019812185) q[2];
cx q[1],q[2];
ry(-1.417282364940057) q[1];
ry(2.378219874923066) q[3];
cx q[1],q[3];
ry(-1.1038083099837337) q[1];
ry(-1.7910265561258045) q[3];
cx q[1],q[3];
ry(-0.7576721039993926) q[2];
ry(1.5349456025583448) q[3];
cx q[2],q[3];
ry(3.0799440970035055) q[2];
ry(1.6921038700143933) q[3];
cx q[2],q[3];
ry(0.7349904854652776) q[0];
ry(0.7872791248909002) q[1];
cx q[0],q[1];
ry(1.3655446458607403) q[0];
ry(-1.5846966001416138) q[1];
cx q[0],q[1];
ry(1.8445291840258378) q[0];
ry(-0.4087416976408057) q[2];
cx q[0],q[2];
ry(1.2164188017569602) q[0];
ry(0.6169750263079283) q[2];
cx q[0],q[2];
ry(-1.6462605490715871) q[0];
ry(-1.4616441861879335) q[3];
cx q[0],q[3];
ry(0.1279354638186377) q[0];
ry(-0.6106799119355267) q[3];
cx q[0],q[3];
ry(-0.19041206106402786) q[1];
ry(3.11689830940927) q[2];
cx q[1],q[2];
ry(1.410799134451152) q[1];
ry(-1.5244034457843956) q[2];
cx q[1],q[2];
ry(-0.06275104175209224) q[1];
ry(1.2949543653970488) q[3];
cx q[1],q[3];
ry(-2.492512612021096) q[1];
ry(-1.4520050586644366) q[3];
cx q[1],q[3];
ry(-0.9282619428527769) q[2];
ry(0.06365061182208898) q[3];
cx q[2],q[3];
ry(1.1754791028439664) q[2];
ry(1.6324090661457067) q[3];
cx q[2],q[3];
ry(-1.5192783556783098) q[0];
ry(-2.6229626288447556) q[1];
cx q[0],q[1];
ry(1.6063718872623503) q[0];
ry(1.6987913339114682) q[1];
cx q[0],q[1];
ry(2.133597876036099) q[0];
ry(0.15310901794124998) q[2];
cx q[0],q[2];
ry(-2.6808933292525476) q[0];
ry(1.4984793638836482) q[2];
cx q[0],q[2];
ry(1.2215099873985) q[0];
ry(-2.269913007162189) q[3];
cx q[0],q[3];
ry(2.006326014862155) q[0];
ry(-2.0988814026840528) q[3];
cx q[0],q[3];
ry(2.0706420046727634) q[1];
ry(0.22735534732451515) q[2];
cx q[1],q[2];
ry(-0.9594479584649298) q[1];
ry(-2.5668928423267157) q[2];
cx q[1],q[2];
ry(1.9240465050947106) q[1];
ry(0.12735728045681327) q[3];
cx q[1],q[3];
ry(-1.650024026935812) q[1];
ry(0.6737876190684879) q[3];
cx q[1],q[3];
ry(-0.7021788137282989) q[2];
ry(-1.4892569472584416) q[3];
cx q[2],q[3];
ry(1.5047061018031402) q[2];
ry(-0.36198012577918864) q[3];
cx q[2],q[3];
ry(1.3270359546723913) q[0];
ry(0.042115104663879924) q[1];
cx q[0],q[1];
ry(-0.49052660507482226) q[0];
ry(1.939829390703266) q[1];
cx q[0],q[1];
ry(-0.22610112988627495) q[0];
ry(-2.409570905632873) q[2];
cx q[0],q[2];
ry(2.6588762115826756) q[0];
ry(-1.7919159528493511) q[2];
cx q[0],q[2];
ry(-0.9474000908033208) q[0];
ry(0.7739371715437566) q[3];
cx q[0],q[3];
ry(0.4839747964445547) q[0];
ry(2.413509538717059) q[3];
cx q[0],q[3];
ry(2.2105548713786374) q[1];
ry(1.5273384996937656) q[2];
cx q[1],q[2];
ry(2.6872969192291456) q[1];
ry(2.248660320170475) q[2];
cx q[1],q[2];
ry(2.8016900086788437) q[1];
ry(-2.138578697878637) q[3];
cx q[1],q[3];
ry(-1.8389569790799083) q[1];
ry(-1.5065878462356102) q[3];
cx q[1],q[3];
ry(-2.846175744642906) q[2];
ry(1.3225567405305991) q[3];
cx q[2],q[3];
ry(-3.0520611253552428) q[2];
ry(0.15753592998185312) q[3];
cx q[2],q[3];
ry(0.014689173025167257) q[0];
ry(1.047987881045118) q[1];
cx q[0],q[1];
ry(-1.94101064652647) q[0];
ry(0.8532318030131254) q[1];
cx q[0],q[1];
ry(2.069786221722291) q[0];
ry(-0.32124292722065206) q[2];
cx q[0],q[2];
ry(-1.9963815746693732) q[0];
ry(-1.116364554635524) q[2];
cx q[0],q[2];
ry(2.2924218107760463) q[0];
ry(-2.4638194015753125) q[3];
cx q[0],q[3];
ry(-0.3595340679304533) q[0];
ry(-2.538007221792374) q[3];
cx q[0],q[3];
ry(1.174340086344297) q[1];
ry(-1.433310109939631) q[2];
cx q[1],q[2];
ry(1.8497733511231238) q[1];
ry(-1.196919724351214) q[2];
cx q[1],q[2];
ry(0.7203409241804524) q[1];
ry(-2.023991363361706) q[3];
cx q[1],q[3];
ry(-2.2556876946991755) q[1];
ry(-0.5006488424561493) q[3];
cx q[1],q[3];
ry(3.0335915879864075) q[2];
ry(1.3000772548028763) q[3];
cx q[2],q[3];
ry(-2.4958157531375003) q[2];
ry(-1.5704199251184345) q[3];
cx q[2],q[3];
ry(-0.32921097710835684) q[0];
ry(-2.683387071591952) q[1];
cx q[0],q[1];
ry(2.8033008303868274) q[0];
ry(2.6855087403237574) q[1];
cx q[0],q[1];
ry(-2.0267052212270347) q[0];
ry(-0.744216871996544) q[2];
cx q[0],q[2];
ry(-1.8587417845430034) q[0];
ry(-1.3297541518554308) q[2];
cx q[0],q[2];
ry(2.936417409496276) q[0];
ry(1.6512262452983226) q[3];
cx q[0],q[3];
ry(0.8626131182810194) q[0];
ry(1.9386170187213176) q[3];
cx q[0],q[3];
ry(1.3817488350251) q[1];
ry(-0.9439592743478449) q[2];
cx q[1],q[2];
ry(-2.4530403730162065) q[1];
ry(1.247910562827547) q[2];
cx q[1],q[2];
ry(-2.452872280999169) q[1];
ry(-0.6962453396051798) q[3];
cx q[1],q[3];
ry(-1.5170828220681762) q[1];
ry(-0.5786321990295002) q[3];
cx q[1],q[3];
ry(1.56343178272183) q[2];
ry(-0.5900334353368234) q[3];
cx q[2],q[3];
ry(-3.120016285922219) q[2];
ry(-2.3432336337373396) q[3];
cx q[2],q[3];
ry(2.4804586702149183) q[0];
ry(1.8489074253525073) q[1];
ry(-1.4702003930418899) q[2];
ry(-1.1760364234131933) q[3];