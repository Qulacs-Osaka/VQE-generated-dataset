OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.2428790310348088) q[0];
rz(0.843452981537661) q[0];
ry(0.5507773056041962) q[1];
rz(-0.37577751107654866) q[1];
ry(-2.6499692899121143) q[2];
rz(2.558783443844628) q[2];
ry(-2.70326193782227) q[3];
rz(-2.9309197663346533) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.6036679830304754) q[0];
rz(-2.7339880829178473) q[0];
ry(0.5859698276787028) q[1];
rz(2.4922607704780546) q[1];
ry(2.2297495692314433) q[2];
rz(0.6762431609174692) q[2];
ry(-0.02040853541175558) q[3];
rz(-1.165499527421218) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.1712152413096861) q[0];
rz(-2.2196053878202933) q[0];
ry(0.27681792850093156) q[1];
rz(-2.665382510588978) q[1];
ry(-2.7076327622781653) q[2];
rz(1.6062753798703977) q[2];
ry(-0.821314003615354) q[3];
rz(-0.5789641647293299) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.7877221499604152) q[0];
rz(-0.5890211919622966) q[0];
ry(-0.050468974469609316) q[1];
rz(-0.2380515431233673) q[1];
ry(1.8791203260218463) q[2];
rz(-2.7562631584036636) q[2];
ry(-0.45441051158731105) q[3];
rz(2.937682202562692) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.4707746284647023) q[0];
rz(1.9180884649292633) q[0];
ry(0.5175260960811495) q[1];
rz(-1.453684217820328) q[1];
ry(2.411586717677356) q[2];
rz(-2.496981856618602) q[2];
ry(-2.8041331119760833) q[3];
rz(2.7418207235027854) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.1423156177038627) q[0];
rz(1.0078125548223038) q[0];
ry(2.664304203167442) q[1];
rz(-2.5525181929224114) q[1];
ry(-1.2449123387264018) q[2];
rz(1.6277762136837488) q[2];
ry(0.38971667464300097) q[3];
rz(3.090227942981134) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.00839976023059241) q[0];
rz(0.8614256259252863) q[0];
ry(-0.39169121366329873) q[1];
rz(1.023054221111198) q[1];
ry(-1.9397717609229668) q[2];
rz(-2.44127360897114) q[2];
ry(-0.16308694767520623) q[3];
rz(-1.9797165948678246) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.9113181855169925) q[0];
rz(-1.118842838121086) q[0];
ry(2.0939822079715333) q[1];
rz(2.167181371802535) q[1];
ry(0.35166976830425195) q[2];
rz(-0.5216456795592972) q[2];
ry(1.82107959849461) q[3];
rz(0.09806693274935574) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-3.051348713143284) q[0];
rz(-0.10868485477766798) q[0];
ry(-1.8987302457412132) q[1];
rz(0.6076660802065655) q[1];
ry(-0.044661878644640766) q[2];
rz(0.7361948181279657) q[2];
ry(-1.5692105712974938) q[3];
rz(-2.308265570192274) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.3702635278799482) q[0];
rz(0.4083006956892547) q[0];
ry(-0.3747390093777574) q[1];
rz(-2.1627350450172935) q[1];
ry(-0.30158381843860127) q[2];
rz(-1.4452650569581522) q[2];
ry(-1.1220816664500495) q[3];
rz(-1.2694517050285752) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-3.074935683555842) q[0];
rz(0.766038965074836) q[0];
ry(0.015683272375200286) q[1];
rz(0.3612181697892405) q[1];
ry(3.1051267165082295) q[2];
rz(-1.646621609239305) q[2];
ry(0.29373451341317) q[3];
rz(-2.277541558932236) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.7613708987061247) q[0];
rz(2.2310677077879273) q[0];
ry(0.41131886671061224) q[1];
rz(-1.7762614282274019) q[1];
ry(2.9292123452290966) q[2];
rz(2.1288251408951266) q[2];
ry(2.9916138268430976) q[3];
rz(1.9756327875130784) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.0292571373780603) q[0];
rz(0.10245600757158879) q[0];
ry(-2.69278266648528) q[1];
rz(0.8889959225166376) q[1];
ry(-2.9723708757087723) q[2];
rz(0.5601730392285991) q[2];
ry(0.06245936115240771) q[3];
rz(1.6507967298866344) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.04371008687669944) q[0];
rz(1.4371277104755678) q[0];
ry(0.2568055210896414) q[1];
rz(3.0866528086328855) q[1];
ry(-2.1660821395063454) q[2];
rz(1.4781372325628772) q[2];
ry(1.0104943174363885) q[3];
rz(-1.1818255616160942) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.949604705634364) q[0];
rz(0.682639382748202) q[0];
ry(2.4733766001046593) q[1];
rz(2.768951669129716) q[1];
ry(-2.8012196545274297) q[2];
rz(-1.877810540844084) q[2];
ry(-0.22996681429061727) q[3];
rz(1.8147541384452506) q[3];