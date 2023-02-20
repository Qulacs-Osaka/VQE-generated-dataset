OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.5831429114130544) q[0];
rz(1.4962299595176545) q[0];
ry(-2.7555714332052315) q[1];
rz(-2.0019251578892274) q[1];
ry(-1.4796426093818145) q[2];
rz(3.141007429157429) q[2];
ry(-0.2634824618973406) q[3];
rz(-3.0698585525858286) q[3];
ry(-1.5731898497217744) q[4];
rz(2.8841862105625626) q[4];
ry(-1.5701070208527144) q[5];
rz(-2.997051769884669) q[5];
ry(-1.3910371961167676) q[6];
rz(-0.01390615498853168) q[6];
ry(1.64993978693533) q[7];
rz(1.7557280586443131) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.36332882059022004) q[0];
rz(3.130112829772254) q[0];
ry(1.5629526304482293) q[1];
rz(-1.5649053730571085) q[1];
ry(-1.6867590863977027) q[2];
rz(1.4804589076476242) q[2];
ry(1.568405053149168) q[3];
rz(1.4762392721316733) q[3];
ry(-1.6738995145139333) q[4];
rz(2.6586304968398307) q[4];
ry(-1.6813501994365998) q[5];
rz(0.015564010571012863) q[5];
ry(-1.5706652051907293) q[6];
rz(0.023357328097437846) q[6];
ry(-3.1264114343086713) q[7];
rz(1.756105326509716) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.07411970096666011) q[0];
rz(0.03830555212103045) q[0];
ry(1.564820549943313) q[1];
rz(1.5659476334970523) q[1];
ry(1.571017328350421) q[2];
rz(-1.6522472630222422) q[2];
ry(1.109905930744648) q[3];
rz(2.088118872994505) q[3];
ry(1.756070308590843) q[4];
rz(3.1319567790247667) q[4];
ry(-1.5813552107586162) q[5];
rz(0.0249377558049968) q[5];
ry(1.4541363550828015) q[6];
rz(3.0878551723327323) q[6];
ry(-1.5715751556873823) q[7];
rz(-0.01520182063026265) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.3318405494017425) q[0];
rz(0.1661697091623999) q[0];
ry(-1.5691043465237773) q[1];
rz(1.7128049633910285) q[1];
ry(-0.049596054839579296) q[2];
rz(-0.8957889501774758) q[2];
ry(0.35453181070438083) q[3];
rz(0.22584843765896068) q[3];
ry(-1.597783198155377) q[4];
rz(-3.140635506907747) q[4];
ry(1.1070887391715925) q[5];
rz(2.434692706118857) q[5];
ry(-2.42306380562625) q[6];
rz(3.1310429668160262) q[6];
ry(-1.5908075373245136) q[7];
rz(1.3877081188821858) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.5672966003861912) q[0];
rz(-0.4620865429861292) q[0];
ry(3.0888480414173283) q[1];
rz(-1.473467977795281) q[1];
ry(-3.091218090923895) q[2];
rz(-2.6413511572752206) q[2];
ry(1.574462156538596) q[3];
rz(0.0017876628851290141) q[3];
ry(-1.834363098899118) q[4];
rz(-0.07613490531753886) q[4];
ry(-3.1116003736203157) q[5];
rz(0.9224831261923577) q[5];
ry(1.5649273670978923) q[6];
rz(-1.238613651466923) q[6];
ry(-3.141314112396972) q[7];
rz(-2.226695578445004) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.851622679341703) q[0];
rz(-1.1975802120605472) q[0];
ry(-1.650161710987157) q[1];
rz(-3.0596907949856433) q[1];
ry(1.573251097598695) q[2];
rz(-1.5683394917366744) q[2];
ry(-1.4855223440357304) q[3];
rz(-0.3547552527332747) q[3];
ry(-1.3418446517354952) q[4];
rz(-0.0043502880153860266) q[4];
ry(-1.0141986552800404) q[5];
rz(1.3166281731421252) q[5];
ry(1.2099522634759214) q[6];
rz(2.329186988240533) q[6];
ry(-2.981968137190378) q[7];
rz(-0.8782308429880241) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-3.141167783375493) q[0];
rz(0.3567295182758299) q[0];
ry(-1.589378787900916) q[1];
rz(1.0478604462166885) q[1];
ry(-1.569446480717966) q[2];
rz(-0.4697933024058649) q[2];
ry(-3.131709901903345) q[3];
rz(1.5646621464568664) q[3];
ry(0.19650491702632333) q[4];
rz(1.4796706696269741) q[4];
ry(3.061890032128091) q[5];
rz(-3.0107426959992982) q[5];
ry(1.5710636466278203) q[6];
rz(2.8431802417005456) q[6];
ry(0.0009309448749702232) q[7];
rz(0.64794653305814) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.8660956990386979) q[0];
rz(0.3522140679195802) q[0];
ry(2.5115759414961745) q[1];
rz(1.4779708652322707) q[1];
ry(1.8694880589468523) q[2];
rz(-2.798016905243031) q[2];
ry(2.0399030059260426) q[3];
rz(-1.0720671449130155) q[3];
ry(-0.38849829359281357) q[4];
rz(0.447947020192446) q[4];
ry(2.0226119908665323) q[5];
rz(-1.1251518461140169) q[5];
ry(-1.750205935956281) q[6];
rz(-1.250222902834934) q[6];
ry(-1.3003758240833174) q[7];
rz(-2.786747988158622) q[7];