OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.603833782725286) q[0];
rz(-2.0183870932669006) q[0];
ry(-1.2189582460542039) q[1];
rz(-3.0757714876048223) q[1];
ry(-0.3261167893924927) q[2];
rz(1.5894224988135086) q[2];
ry(2.377467984012511) q[3];
rz(-1.0064295283892843) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.218692997702491) q[0];
rz(1.7739559604275037) q[0];
ry(-2.9610154759005027) q[1];
rz(-2.3665532248216543) q[1];
ry(-2.7196864470841997) q[2];
rz(-1.1175092297063616) q[2];
ry(0.4723154141511134) q[3];
rz(-2.0017626330296636) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.5638453979931612) q[0];
rz(0.595592967870291) q[0];
ry(0.8240840723342003) q[1];
rz(1.6390261413791214) q[1];
ry(1.152838285909386) q[2];
rz(3.1377472994081996) q[2];
ry(2.2673475284126106) q[3];
rz(-1.3088115517277041) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.431372907002952) q[0];
rz(-1.956371272169697) q[0];
ry(-1.5256657271523646) q[1];
rz(-0.8174101595631069) q[1];
ry(-0.24121401865872147) q[2];
rz(2.19091516292414) q[2];
ry(2.533271283881753) q[3];
rz(0.8090635515185128) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.573138187100893) q[0];
rz(-1.6433625247554104) q[0];
ry(-1.7439197070275902) q[1];
rz(2.213529666290988) q[1];
ry(0.9965209078251419) q[2];
rz(2.730364571055089) q[2];
ry(-1.1214548932006085) q[3];
rz(-1.7566715722664983) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.7032884970723603) q[0];
rz(-3.008593405886892) q[0];
ry(1.8024420737444407) q[1];
rz(-1.9818835072891734) q[1];
ry(-0.9010029637828935) q[2];
rz(1.4694881503998551) q[2];
ry(-1.4318552821554589) q[3];
rz(-1.5995950533922296) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.020918311428904) q[0];
rz(0.7063941166279077) q[0];
ry(0.8717030314034675) q[1];
rz(-2.391593396628015) q[1];
ry(2.736535651450467) q[2];
rz(0.6459999186145162) q[2];
ry(0.8185614653949448) q[3];
rz(-2.5434664202713444) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(3.051635051816417) q[0];
rz(1.9654280642932056) q[0];
ry(-2.1285764100395235) q[1];
rz(-2.4865352465138932) q[1];
ry(2.5321590171758785) q[2];
rz(-2.1153861814735526) q[2];
ry(-1.0162146315091052) q[3];
rz(1.3362315373828704) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.8275123986069373) q[0];
rz(2.9198514057048075) q[0];
ry(1.4280356732437784) q[1];
rz(2.576678995067901) q[1];
ry(1.2107966187988852) q[2];
rz(-2.7331845822402565) q[2];
ry(-2.9741643265383684) q[3];
rz(1.0561186238623046) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.530736341283532) q[0];
rz(-0.8903427427280439) q[0];
ry(2.3297077402143618) q[1];
rz(0.8114534238160752) q[1];
ry(2.5915828831173395) q[2];
rz(2.7965559161029088) q[2];
ry(-1.673265498592649) q[3];
rz(2.6334492382541486) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.4754631089885764) q[0];
rz(0.10635621123963722) q[0];
ry(2.1151375999696675) q[1];
rz(2.4954003152978466) q[1];
ry(-1.7456533636037943) q[2];
rz(0.9377429394132513) q[2];
ry(-1.611598847108243) q[3];
rz(2.3898054746990733) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.0186513578509873) q[0];
rz(2.834902242126552) q[0];
ry(-0.7160605411961057) q[1];
rz(0.6093836447461415) q[1];
ry(1.2077309287756317) q[2];
rz(2.315348523820822) q[2];
ry(-2.581629375700466) q[3];
rz(2.5304830688327047) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.516591604047318) q[0];
rz(1.2401117904885837) q[0];
ry(-0.8986861242945475) q[1];
rz(2.7814110053041183) q[1];
ry(-0.046202234615708215) q[2];
rz(2.7248498784706214) q[2];
ry(2.332439005727799) q[3];
rz(-0.9216010397304553) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.80874761989626) q[0];
rz(-1.7986746431301919) q[0];
ry(-0.6176214195439002) q[1];
rz(1.0986818613117295) q[1];
ry(1.7494459004825862) q[2];
rz(-1.2914757817501483) q[2];
ry(-0.9817202687293278) q[3];
rz(-2.475399488102823) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.8064492675079977) q[0];
rz(2.4180377397222768) q[0];
ry(0.07297366634296218) q[1];
rz(-1.4609655163095479) q[1];
ry(-2.5141319747499575) q[2];
rz(-0.7898662255116582) q[2];
ry(-0.0968532684990171) q[3];
rz(-1.6587138127652878) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.3127865345743759) q[0];
rz(-0.6547623844779165) q[0];
ry(1.251284012266932) q[1];
rz(0.017121195378030585) q[1];
ry(-1.1902056080188574) q[2];
rz(-0.6610842757805306) q[2];
ry(-1.6552041629672485) q[3];
rz(2.6978174342914807) q[3];