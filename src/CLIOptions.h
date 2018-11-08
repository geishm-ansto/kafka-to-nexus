#pragma once

#include <string>

// Forward declarations
namespace CLI {
class App;
class Option;
}
namespace uri {
class URI;
}
struct MainOpt;

void setCLIOptions(CLI::App &App, MainOpt &MainOptions);

CLI::Option *addOption(CLI::App &App, std::string Name, uri::URI &URIArg,
                       bool &TrueIfOptionGiven, std::string Description,
                       bool Defaulted);

/// Use for adding a URI option
CLI::Option *addOption(CLI::App &App, std::string Name, uri::URI &URIArg,
                       std::string Description, bool Defaulted);
