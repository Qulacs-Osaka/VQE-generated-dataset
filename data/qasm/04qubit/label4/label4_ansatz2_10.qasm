OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.4878304149753867) q[0];
rz(-1.7013131120426142) q[0];
ry(1.63099694545296) q[1];
rz(1.8733742773446613) q[1];
ry(-0.7069170940483421) q[2];
rz(1.0721675781315643) q[2];
ry(-2.076572000830752) q[3];
rz(-1.8698485794747741) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.615962957067482) q[0];
rz(1.9579353240511181) q[0];
ry(0.4733944566418331) q[1];
rz(3.0810538571908213) q[1];
ry(-2.8111641869198496) q[2];
rz(-2.3627432701434183) q[2];
ry(2.8204065819783763) q[3];
rz(1.7288242905594382) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.2341199283027122) q[0];
rz(1.7484099043287948) q[0];
ry(2.5688376786267173) q[1];
rz(-0.6595803494572269) q[1];
ry(-1.9233297181807973) q[2];
rz(-1.08478196793902) q[2];
ry(-1.0857302719231683) q[3];
rz(1.5480774156010089) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-3.1060034653505677) q[0];
rz(-0.8885722962620749) q[0];
ry(0.45651755030996366) q[1];
rz(-0.09881062363680793) q[1];
ry(2.478879811271622) q[2];
rz(-2.7502053156051156) q[2];
ry(0.3617276127619835) q[3];
rz(-0.21876393782543202) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.1503729043134052) q[0];
rz(-0.6475833025877022) q[0];
ry(-2.655118290109896) q[1];
rz(1.5119672315245447) q[1];
ry(1.2429470599904962) q[2];
rz(0.12683717634222802) q[2];
ry(-1.7514295005390146) q[3];
rz(1.4575635169918042) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.8887273271563405) q[0];
rz(-2.3926038339765014) q[0];
ry(2.8964072608951916) q[1];
rz(2.4753218522962834) q[1];
ry(0.37697924254553694) q[2];
rz(2.403224938785008) q[2];
ry(1.0582880296320107) q[3];
rz(-1.327448697908433) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.845252157002686) q[0];
rz(1.2140338849210206) q[0];
ry(-0.8963739514020738) q[1];
rz(0.9871061306794635) q[1];
ry(-1.3981725295580871) q[2];
rz(1.717808176431281) q[2];
ry(2.012792202697042) q[3];
rz(-0.948178891449932) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.8157130412634115) q[0];
rz(-2.044681382604262) q[0];
ry(0.19328755095829053) q[1];
rz(-0.14801235201083077) q[1];
ry(-1.9612972020568762) q[2];
rz(3.1015745996724204) q[2];
ry(-1.9356895394225027) q[3];
rz(1.3694517979495782) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.33317025748819223) q[0];
rz(-0.6130538850398448) q[0];
ry(0.6858678155090726) q[1];
rz(-1.8068506794360077) q[1];
ry(2.3618205028618826) q[2];
rz(2.1199586069388108) q[2];
ry(0.4802259075430018) q[3];
rz(-0.334118416435973) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.136692882182577) q[0];
rz(1.2705257067888427) q[0];
ry(-2.9510438947186404) q[1];
rz(-2.62725737200587) q[1];
ry(-0.6123826234396533) q[2];
rz(1.680492471434015) q[2];
ry(-2.0453866521375983) q[3];
rz(-1.1878345891351652) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.49898928519627) q[0];
rz(-3.118280935863371) q[0];
ry(-1.4936371097494439) q[1];
rz(-0.23820940739368476) q[1];
ry(-1.097055666138371) q[2];
rz(2.265314612463105) q[2];
ry(1.012157378347661) q[3];
rz(-1.6892176763956135) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.27059739442281944) q[0];
rz(-2.66981545753512) q[0];
ry(-0.35492086768232894) q[1];
rz(-1.512194773707936) q[1];
ry(-0.5845108426924339) q[2];
rz(-0.8660053770441092) q[2];
ry(1.3012384279107276) q[3];
rz(1.6601630323284837) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.53183577640034) q[0];
rz(1.3453475719666512) q[0];
ry(0.3992985403738398) q[1];
rz(-0.06367393198202151) q[1];
ry(-2.9842099928652996) q[2];
rz(-2.580271881376994) q[2];
ry(1.6371049804954905) q[3];
rz(1.535352559902865) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.0939471140210753) q[0];
rz(1.1812827797005585) q[0];
ry(-2.3506620028102168) q[1];
rz(-0.7833507885948525) q[1];
ry(2.362712679584206) q[2];
rz(-2.006604504157445) q[2];
ry(0.750204233780531) q[3];
rz(-1.6465239996962644) q[3];