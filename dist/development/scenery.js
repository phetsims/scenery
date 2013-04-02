(function() {

;(function(g){

    // summary: A simple feature detection function/framework.
    //
    // name: String
    //      The name of the feature to detect, as defined by the overall `has` tests.
    //      Tests can be registered via `has.add(testname, testfunction)`.
    //
    // example:
    //      mylibrary.bind = has("native-bind") ? function(fn, context){
    //          return fn.bind(context);
    //      } : function(fn, context){
    //          return function(){
    //              fn.apply(context, arguments);
    //          }
    //      }

    var NON_HOST_TYPES = { "boolean": 1, "number": 1, "string": 1, "undefined": 1 },
        VENDOR_PREFIXES = ["Webkit", "Moz", "O", "ms", "Khtml"],
        d = isHostType(g, "document") && g.document,
        el = d && isHostType(d, "createElement") && d.createElement("DiV"),
        freeExports = typeof exports == "object" && exports,
        freeModule = typeof module == "object" && module,
        testCache = {}
    ;

    function has(/* String */name){
        if(typeof testCache[name] == "function"){
            testCache[name] = testCache[name](g, d, el);
        }
        return testCache[name]; // Boolean
    }

    function add(/* String */name, /* Function */test, /* Boolean? */now){
        // summary: Register a new feature detection test for some named feature
        //
        // name: String
        //      The name of the feature to test.
        //
        // test: Function
        //      A test function to register. If a function, queued for testing until actually
        //      needed. The test function should return a boolean indicating
        //      the presence of a feature or bug.
        //
        // now: Boolean?
        //      Optional. Omit if `test` is not a function. Provides a way to immediately
        //      run the test and cache the result.
        // example:
        //      A redundant test, testFn with immediate execution:
        //  |       has.add("javascript", function(){ return true; }, true);
        //
        // example:
        //      Again with the redundantness. You can do this in your tests, but we should
        //      not be doing this in any internal has.js tests
        //  |       has.add("javascript", true);
        //
        // example:
        //      Three things are passed to the testFunction. `global`, `document`, and a generic element
        //      from which to work your test should the need arise.
        //  |       has.add("bug-byid", function(g, d, el){
        //  |           // g  == global, typically window, yadda yadda
        //  |           // d  == document object
        //  |           // el == the generic element. a `has` element.
        //  |           return false; // fake test, byid-when-form-has-name-matching-an-id is slightly longer
        //  |       });
        testCache[name] = now ? test(g, d, el) : test;
    }

    // cssprop adapted from http://gist.github.com/598008 (thanks, ^pi)
    function cssprop(name, el){
        var supported = false,
            capitalized = name.charAt(0).toUpperCase() + name.slice(1),
            length = VENDOR_PREFIXES.length,
            style = el.style;

        if(typeof style[name] == "string"){
            supported = true;
        }else{
            while(length--){
                if(typeof style[VENDOR_PREFIXES[length] + capitalized] == "string"){
                    supported = true;
                    break;
                }
            }
        }
        return supported;
    }

    function clearElement(el){
        if(el){
            while(el.lastChild){
                el.removeChild(el.lastChild);
            }
        }
        return el;
    }

    // Host objects can return type values that are different from their actual
    // data type. The objects we are concerned with usually return non-primitive
    // types of object, function, or unknown.
    function isHostType(object, property){
        var type = typeof object[property];
        return type == "object" ? !!object[property] : !NON_HOST_TYPES[type];
    }

        has.add = add;
    has.clearElement = clearElement;
    has.cssprop = cssprop;
    has.isHostType = isHostType;
    has._tests = testCache;

    has.add("dom", function(g, d, el){
        return d && el && isHostType(g, "location") && isHostType(d, "documentElement") &&
            isHostType(d, "getElementById") && isHostType(d, "getElementsByName") &&
            isHostType(d, "getElementsByTagName") && isHostType(d, "createComment") &&
            isHostType(d, "createElement") && isHostType(d, "createTextNode") &&
            isHostType(el, "appendChild") && isHostType(el, "insertBefore") &&
            isHostType(el, "removeChild") && isHostType(el, "getAttribute") &&
            isHostType(el, "setAttribute") && isHostType(el, "removeAttribute") &&
            isHostType(el, "style") && typeof el.style.cssText == "string";
    });

    // Stop repeat background-image requests and reduce memory consumption in IE6 SP1
    // http://misterpixel.blogspot.com/2006/09/forensic-analysis-of-ie6.html
    // http://blogs.msdn.com/b/cwilso/archive/2006/11/07/ie-re-downloading-background-images.aspx?PageIndex=1
    // http://support.microsoft.com/kb/823727
    try{
        document.execCommand("BackgroundImageCache", false, true);
    }catch(e){}

    // Expose has()
    // some AMD build optimizers, like r.js, check for specific condition patterns like the following:
    if(typeof define == "function" && typeof define.amd == "object" && define.amd){
        define("has", function(){
            return has;
        });
    }
    // check for `exports` after `define` in case a build optimizer adds an `exports` object
    else if(freeExports){
        // in Node.js or RingoJS v0.8.0+
        if(freeModule && freeModule.exports == freeExports){
          (freeModule.exports = has).has = has;
        }
        // in Narwhal or RingoJS v0.7.0-
        else{
          freeExports.has = has;
        }
    }
    // in a browser or Rhino
    else{
        // use square bracket notation so Closure Compiler won't munge `has`
        // http://code.google.com/closure/compiler/docs/api-tutorial3.html#export
        g["has"] = has;
    }
})(this);

/**
 * almond 0.2.5 Copyright (c) 2011-2012, The Dojo Foundation All Rights Reserved.
 * Available via the MIT or new BSD license.
 * see: http://github.com/jrburke/almond for details
 */
//Going sloppy to avoid 'use strict' string cost, but strict practices should
//be followed.
/*jslint sloppy: true */
/*global setTimeout: false */

var requirejs, require, define;
(function (undef) {
    var main, req, makeMap, handlers,
        defined = {},
        waiting = {},
        config = {},
        defining = {},
        hasOwn = Object.prototype.hasOwnProperty,
        aps = [].slice;

    function hasProp(obj, prop) {
        return hasOwn.call(obj, prop);
    }

    /**
     * Given a relative module name, like ./something, normalize it to
     * a real name that can be mapped to a path.
     * @param {String} name the relative name
     * @param {String} baseName a real name that the name arg is relative
     * to.
     * @returns {String} normalized name
     */
    function normalize(name, baseName) {
        var nameParts, nameSegment, mapValue, foundMap,
            foundI, foundStarMap, starI, i, j, part,
            baseParts = baseName && baseName.split("/"),
            map = config.map,
            starMap = (map && map['*']) || {};

        //Adjust any relative paths.
        if (name && name.charAt(0) === ".") {
            //If have a base name, try to normalize against it,
            //otherwise, assume it is a top-level require that will
            //be relative to baseUrl in the end.
            if (baseName) {
                //Convert baseName to array, and lop off the last part,
                //so that . matches that "directory" and not name of the baseName's
                //module. For instance, baseName of "one/two/three", maps to
                //"one/two/three.js", but we want the directory, "one/two" for
                //this normalization.
                baseParts = baseParts.slice(0, baseParts.length - 1);

                name = baseParts.concat(name.split("/"));

                //start trimDots
                for (i = 0; i < name.length; i += 1) {
                    part = name[i];
                    if (part === ".") {
                        name.splice(i, 1);
                        i -= 1;
                    } else if (part === "..") {
                        if (i === 1 && (name[2] === '..' || name[0] === '..')) {
                            //End of the line. Keep at least one non-dot
                            //path segment at the front so it can be mapped
                            //correctly to disk. Otherwise, there is likely
                            //no path mapping for a path starting with '..'.
                            //This can still fail, but catches the most reasonable
                            //uses of ..
                            break;
                        } else if (i > 0) {
                            name.splice(i - 1, 2);
                            i -= 2;
                        }
                    }
                }
                //end trimDots

                name = name.join("/");
            } else if (name.indexOf('./') === 0) {
                // No baseName, so this is ID is resolved relative
                // to baseUrl, pull off the leading dot.
                name = name.substring(2);
            }
        }

        //Apply map config if available.
        if ((baseParts || starMap) && map) {
            nameParts = name.split('/');

            for (i = nameParts.length; i > 0; i -= 1) {
                nameSegment = nameParts.slice(0, i).join("/");

                if (baseParts) {
                    //Find the longest baseName segment match in the config.
                    //So, do joins on the biggest to smallest lengths of baseParts.
                    for (j = baseParts.length; j > 0; j -= 1) {
                        mapValue = map[baseParts.slice(0, j).join('/')];

                        //baseName segment has  config, find if it has one for
                        //this name.
                        if (mapValue) {
                            mapValue = mapValue[nameSegment];
                            if (mapValue) {
                                //Match, update name to the new value.
                                foundMap = mapValue;
                                foundI = i;
                                break;
                            }
                        }
                    }
                }

                if (foundMap) {
                    break;
                }

                //Check for a star map match, but just hold on to it,
                //if there is a shorter segment match later in a matching
                //config, then favor over this star map.
                if (!foundStarMap && starMap && starMap[nameSegment]) {
                    foundStarMap = starMap[nameSegment];
                    starI = i;
                }
            }

            if (!foundMap && foundStarMap) {
                foundMap = foundStarMap;
                foundI = starI;
            }

            if (foundMap) {
                nameParts.splice(0, foundI, foundMap);
                name = nameParts.join('/');
            }
        }

        return name;
    }

    function makeRequire(relName, forceSync) {
        return function () {
            //A version of a require function that passes a moduleName
            //value for items that may need to
            //look up paths relative to the moduleName
            return req.apply(undef, aps.call(arguments, 0).concat([relName, forceSync]));
        };
    }

    function makeNormalize(relName) {
        return function (name) {
            return normalize(name, relName);
        };
    }

    function makeLoad(depName) {
        return function (value) {
            defined[depName] = value;
        };
    }

    function callDep(name) {
        if (hasProp(waiting, name)) {
            var args = waiting[name];
            delete waiting[name];
            defining[name] = true;
            main.apply(undef, args);
        }

        if (!hasProp(defined, name) && !hasProp(defining, name)) {
            throw new Error('No ' + name);
        }
        return defined[name];
    }

    //Turns a plugin!resource to [plugin, resource]
    //with the plugin being undefined if the name
    //did not have a plugin prefix.
    function splitPrefix(name) {
        var prefix,
            index = name ? name.indexOf('!') : -1;
        if (index > -1) {
            prefix = name.substring(0, index);
            name = name.substring(index + 1, name.length);
        }
        return [prefix, name];
    }

    /**
     * Makes a name map, normalizing the name, and using a plugin
     * for normalization if necessary. Grabs a ref to plugin
     * too, as an optimization.
     */
    makeMap = function (name, relName) {
        var plugin,
            parts = splitPrefix(name),
            prefix = parts[0];

        name = parts[1];

        if (prefix) {
            prefix = normalize(prefix, relName);
            plugin = callDep(prefix);
        }

        //Normalize according
        if (prefix) {
            if (plugin && plugin.normalize) {
                name = plugin.normalize(name, makeNormalize(relName));
            } else {
                name = normalize(name, relName);
            }
        } else {
            name = normalize(name, relName);
            parts = splitPrefix(name);
            prefix = parts[0];
            name = parts[1];
            if (prefix) {
                plugin = callDep(prefix);
            }
        }

        //Using ridiculous property names for space reasons
        return {
            f: prefix ? prefix + '!' + name : name, //fullName
            n: name,
            pr: prefix,
            p: plugin
        };
    };

    function makeConfig(name) {
        return function () {
            return (config && config.config && config.config[name]) || {};
        };
    }

    handlers = {
        require: function (name) {
            return makeRequire(name);
        },
        exports: function (name) {
            var e = defined[name];
            if (typeof e !== 'undefined') {
                return e;
            } else {
                return (defined[name] = {});
            }
        },
        module: function (name) {
            return {
                id: name,
                uri: '',
                exports: defined[name],
                config: makeConfig(name)
            };
        }
    };

    main = function (name, deps, callback, relName) {
        var cjsModule, depName, ret, map, i,
            args = [],
            usingExports;

        //Use name if no relName
        relName = relName || name;

        //Call the callback to define the module, if necessary.
        if (typeof callback === 'function') {

            //Pull out the defined dependencies and pass the ordered
            //values to the callback.
            //Default to [require, exports, module] if no deps
            deps = !deps.length && callback.length ? ['require', 'exports', 'module'] : deps;
            for (i = 0; i < deps.length; i += 1) {
                map = makeMap(deps[i], relName);
                depName = map.f;

                //Fast path CommonJS standard dependencies.
                if (depName === "require") {
                    args[i] = handlers.require(name);
                } else if (depName === "exports") {
                    //CommonJS module spec 1.1
                    args[i] = handlers.exports(name);
                    usingExports = true;
                } else if (depName === "module") {
                    //CommonJS module spec 1.1
                    cjsModule = args[i] = handlers.module(name);
                } else if (hasProp(defined, depName) ||
                           hasProp(waiting, depName) ||
                           hasProp(defining, depName)) {
                    args[i] = callDep(depName);
                } else if (map.p) {
                    map.p.load(map.n, makeRequire(relName, true), makeLoad(depName), {});
                    args[i] = defined[depName];
                } else {
                    throw new Error(name + ' missing ' + depName);
                }
            }

            ret = callback.apply(defined[name], args);

            if (name) {
                //If setting exports via "module" is in play,
                //favor that over return value and exports. After that,
                //favor a non-undefined return value over exports use.
                if (cjsModule && cjsModule.exports !== undef &&
                        cjsModule.exports !== defined[name]) {
                    defined[name] = cjsModule.exports;
                } else if (ret !== undef || !usingExports) {
                    //Use the return value from the function.
                    defined[name] = ret;
                }
            }
        } else if (name) {
            //May just be an object definition for the module. Only
            //worry about defining if have a module name.
            defined[name] = callback;
        }
    };

    requirejs = require = req = function (deps, callback, relName, forceSync, alt) {
        if (typeof deps === "string") {
            if (handlers[deps]) {
                //callback in this case is really relName
                return handlers[deps](callback);
            }
            //Just return the module wanted. In this scenario, the
            //deps arg is the module name, and second arg (if passed)
            //is just the relName.
            //Normalize module name, if it contains . or ..
            return callDep(makeMap(deps, callback).f);
        } else if (!deps.splice) {
            //deps is a config object, not an array.
            config = deps;
            if (callback.splice) {
                //callback is an array, which means it is a dependency list.
                //Adjust args if there are dependencies
                deps = callback;
                callback = relName;
                relName = null;
            } else {
                deps = undef;
            }
        }

        //Support require(['a'])
        callback = callback || function () {};

        //If relName is a function, it is an errback handler,
        //so remove it.
        if (typeof relName === 'function') {
            relName = forceSync;
            forceSync = alt;
        }

        //Simulate async callback;
        if (forceSync) {
            main(undef, deps, callback, relName);
        } else {
            //Using a non-zero value because of concern for what old browsers
            //do, and latest browsers "upgrade" to 4 if lower value is used:
            //http://www.whatwg.org/specs/web-apps/current-work/multipage/timers.html#dom-windowtimers-settimeout:
            //If want a value immediately, use require('id') instead -- something
            //that works in almond on the global level, but not guaranteed and
            //unlikely to work in other AMD implementations.
            setTimeout(function () {
                main(undef, deps, callback, relName);
            }, 4);
        }

        return req;
    };

    /**
     * Just drops the config on the floor, but returns req in case
     * the config return value is used.
     */
    req.config = function (cfg) {
        config = cfg;
        if (config.deps) {
            req(config.deps, config.callback);
        }
        return req;
    };

    define = function (name, deps, callback) {

        //This module may not have dependencies
        if (!deps.splice) {
            //deps is not an array, so probably means
            //an object literal or factory function for
            //the value. Adjust args.
            callback = deps;
            deps = [];
        }

        if (!hasProp(defined, name) && !hasProp(waiting, name)) {
            waiting[name] = [name, deps, callback];
        }
    };

    define.amd = {
        jQuery: true
    };
}());

define("almond", function(){});

// Copyright 2002-2012, University of Colorado

/**
 * The main 'scenery' namespace object for the exported (non-Require.js) API. Used internally
 * since it prevents Require.js issues with circular dependencies.
 *
 * The returned scenery object namespace may be incomplete if not all modules are listed as
 * dependencies. Please use the 'main' module for that purpose if all of Scenery is desired.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/scenery',['require'], function( require ) {
  // will be filled in by other modules
  return {};
} );

// Copyright 2002-2012, University of Colorado

/*
 * Usage:
 * var assert = require( '<assert>' )( 'flagName' );
 *
 * assert && assert( <simple value or big computation>, "<message here>" );
 *
 * TODO: decide on usages and viability, and if so document further
 *
 * NOTE: for changing build, add has.js tests for 'assert.' + flagName
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('ASSERT/assert',['require'], function( require ) {
  var assert = function( name, excludeByDefault ) {
    var hasName = 'assert.' + name;
    
    var flagDefined = window.has && window.has( hasName ) !== undefined;
    var skipAssert = flagDefined ? !window.has( hasName ) : excludeByDefault;
    
    if ( skipAssert ) {
      return null;
    } else {
      return function( predicate, message ) {
        var result = typeof predicate === 'function' ? predicate() : predicate;
        
        if ( !result ) {
          // TODO: custom error?
          throw new Error( 'Assertion failed: ' + message );
        }
      };
    }
  };
  
  return assert;
} );

// Copyright 2002-2012, University of Colorado

/**
 * A debugging version of the CanvasRenderingContext2D that will output all commands issued,
 * but can also forward them to a real context
 *
 * See the spec at http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#2dcontext
 * Wrapping of the CanvasRenderingContext2D interface as of January 27th, 2013 (but not other interfaces like TextMetrics and Path)
 *
 * Shortcut to create:
 *    var context = new scenery.DebugContext( document.createElement( 'canvas' ).getContext( '2d' ) );
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/debug/DebugContext',['require','ASSERT/assert','SCENERY/scenery'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  // used to serialize arguments so that it displays exactly like the call would be executed
  function s( value ) {
    return JSON.stringify( value );
  }
  
  function log( message ) {
    console.log( 'context.' + message + ';' );
  }
  
  function attributeGet( name ) {
    log( name );
  }
  
  function attributeSet( name, value ) {
    log( name + ' = ' + s( value ) );
  }
  
  function command( name, args ) {
    if ( args === undefined || args.length === 0 ) {
      log( name + '()' );
    } else {
      log( name + '( ' + _.reduce( args, function( memo, arg ) {
        if ( memo.length > 0 ) {
          return memo + ', ' + s( arg );
        } else {
          return s( arg );
        }
      }, '' ) + ' )' );
    }
  }
  
  scenery.DebugContext = function( context ) {
    this._context = context;
    
    // allow checking of context.ellipse for existence
    if ( context && !context.ellipse ) {
      this.ellipse = context.ellipse;
    }
  };
  var DebugContext = scenery.DebugContext;
  
  DebugContext.prototype = {
    constructor: DebugContext,
    
    get canvas() {
      attributeGet( 'canvas' );
      return this._context.canvas;
    },
    
    get width() {
      attributeGet( 'width' );
      return this._context.width;
    },
    
    get height() {
      attributeGet( 'height' );
      return this._context.height;
    },
    
    commit: function() {
      command( 'commit' );
      this._context.commit();
    },
    
    save: function() {
      command( 'save' );
      this._context.save();
    },
    
    restore: function() {
      command( 'restore' );
      this._context.restore();
    },
    
    get currentTransform() {
      attributeGet( 'currentTransform' );
      return this._context.currentTransform;
    },
    
    set currentTransform( transform ) {
      attributeSet( 'currentTransform', transform );
      this._context.currentTransform = transform;
    },
    
    scale: function( x, y ) {
      command( 'scale', [ x, y ] );
      this._context.scale( x, y );
    },
    
    rotate: function( angle ) {
      command( 'rotate', [ angle ] );
      this._context.rotate( angle );
    },
    
    translate: function( x, y ) {
      command( 'translate', [ x, y ] );
      this._context.translate( x, y );
    },
    
    transform: function( a, b, c, d, e, f ) {
      command( 'transform', [ a, b, c, d, e, f ] );
      this._context.transform( a, b, c, d, e, f );
    },
    
    setTransform: function( a, b, c, d, e, f ) {
      command( 'setTransform', [ a, b, c, d, e, f ] );
      this._context.setTransform( a, b, c, d, e, f );
    },
    
    resetTransform: function() {
      command( 'resetTransform' );
      this._context.resetTransform();
    },
    
    get globalAlpha() {
      attributeGet( 'globalAlpha' );
      return this._context.globalAlpha;
    },
    
    set globalAlpha( value ) {
      attributeSet( 'globalAlpha', value );
      this._context.globalAlpha = value;
    },
    
    get globalCompositeOperation() {
      attributeGet( 'globalCompositeOperation' );
      return this._context.globalCompositeOperation;
    },
    
    set globalCompositeOperation( value ) {
      attributeSet( 'globalCompositeOperation', value );
      this._context.globalCompositeOperation = value;
    },
    
    get imageSmoothingEnabled() {
      attributeGet( 'imageSmoothingEnabled' );
      return this._context.imageSmoothingEnabled;
    },
    
    set imageSmoothingEnabled( value ) {
      attributeSet( 'imageSmoothingEnabled', value );
      this._context.imageSmoothingEnabled = value;
    },
    
    get strokeStyle() {
      attributeGet( 'strokeStyle' );
      return this._context.strokeStyle;
    },
    
    set strokeStyle( value ) {
      attributeSet( 'strokeStyle', value );
      this._context.strokeStyle = value;
    },
    
    get fillStyle() {
      attributeGet( 'fillStyle' );
      return this._context.fillStyle;
    },
    
    set fillStyle( value ) {
      attributeSet( 'fillStyle', value );
      this._context.fillStyle = value;
    },
    
    createLinearGradient: function( x0, y0, x1, y1 ) {
      command( 'createLinearGradient', [ x0, y0, x1, y1 ] );
      return this._context.createLinearGradient( x0, y0, x1, y1 );
    },
    
    createRadialGradient: function( x0, y0, r0, x1, y1, r1 ) {
      command( 'createRadialGradient', [ x0, y0, r0, x1, y1, r1 ] );
      return this._context.createRadialGradient( x0, y0, r0, x1, y1, r1 );
    },
    
    createPattern: function( image, repetition ) {
      command( 'createPattern', [ image, repetition ] );
      return this._context.createPattern( image, repetition );
    },
    
    get shadowOffsetX() {
      attributeGet( 'shadowOffsetX' );
      return this._context.shadowOffsetX;
    },
    
    set shadowOffsetX( value ) {
      attributeSet( 'shadowOffsetX', value );
      this._context.shadowOffsetX = value;
    },
    
    get shadowOffsetY() {
      attributeGet( 'shadowOffsetY' );
      return this._context.shadowOffsetY;
    },
    
    set shadowOffsetY( value ) {
      attributeSet( 'shadowOffsetY', value );
      this._context.shadowOffsetY = value;
    },
    
    get shadowBlur() {
      attributeGet( 'shadowBlur' );
      return this._context.shadowBlur;
    },
    
    set shadowBlur( value ) {
      attributeSet( 'shadowBlur', value );
      this._context.shadowBlur = value;
    },
    
    get shadowColor() {
      attributeGet( 'shadowColor' );
      return this._context.shadowColor;
    },
    
    set shadowColor( value ) {
      attributeSet( 'shadowColor', value );
      this._context.shadowColor = value;
    },
    
    clearRect: function( x, y, w, h ) {
      command( 'clearRect', [ x, y, w, h ] );
      this._context.clearRect( x, y, w, h );
    },
    
    fillRect: function( x, y, w, h ) {
      command( 'fillRect', [ x, y, w, h ] );
      this._context.fillRect( x, y, w, h );
    },
    
    strokeRect: function( x, y, w, h ) {
      command( 'strokeRect', [ x, y, w, h ] );
      this._context.strokeRect( x, y, w, h );
    },
    
    get fillRule() {
      attributeGet( 'fillRule' );
      return this._context.fillRule;
    },
    
    set fillRule( value ) {
      attributeSet( 'fillRule', value );
      this._context.fillRule = value;
    },
    
    beginPath: function() {
      command( 'beginPath' );
      this._context.beginPath();
    },
    
    fill: function( path ) {
      command( 'fill', path ? [ path ] : undefined );
      this._context.fill( path );
    },
    
    stroke: function( path ) {
      command( 'stroke', path ? [ path ] : undefined );
      this._context.stroke( path );
    },
    
    drawSystemFocusRing: function( a, b ) {
      command( 'drawSystemFocusRing', b ? [ a, b ] : [ a ] );
      this._context.drawSystemFocusRing( a, b );
    },
    
    drawCustomFocusRing: function( a, b ) {
      command( 'drawCustomFocusRing', b ? [ a, b ] : [ a ] );
      return this._context.drawCustomFocusRing( a, b );
    },
    
    scrollPathIntoView: function( path ) {
      command( 'scrollPathIntoView', path ? [ path ] : undefined );
      this._context.scrollPathIntoView( path );
    },
    
    clip: function( path ) {
      command( 'clip', path ? [ path ] : undefined );
      this._context.clip( path );
    },
    
    resetClip: function() {
      command( 'resetClip' );
      this._context.resetClip();
    },
    
    isPointInPath: function( a, b, c ) {
      command( 'isPointInPath', c ? [ a, b, c ] : [ a, b ] );
      return this._context.isPointInPath( a, b, c );
    },
    
    fillText: function( text, x, y, maxWidth ) {
      command( 'fillText', maxWidth !== undefined ? [ text, x, y, maxWidth ] : [ text, x, y ] );
      this._context.fillText( text, x, y, maxWidth );
    },
    
    strokeText: function( text, x, y, maxWidth ) {
      command( 'strokeText', maxWidth !== undefined ? [ text, x, y, maxWidth ] : [ text, x, y ] );
      this._context.strokeText( text, x, y, maxWidth );
    },
    
    measureText: function( text ) {
      command( 'measureText', [ text ] );
      return this._context.measureText( text );
    },
    
    drawImage: function( image, a, b, c, d, e, f, g, h ) {
      command( 'drawImage', c !== undefined ? ( e !== undefined ? [ image, a, b, c, d, e, f, g, h ] : [ image, a, b, c, d ] ) : [ image, a, b ] );
      this._context.drawImage( image, a, b, c, d, e, f, g, h );
    },
    
    addHitRegion: function( options ) {
      command( 'addHitRegion', [ options ] );
      this._context.addHitRegion( options );
    },
    
    removeHitRegion: function( options ) {
      command( 'removeHitRegion', [ options ] );
      this._context.removeHitRegion( options );
    },
    
    createImageData: function( a, b ) {
      command( 'createImageData', b !== undefined ? [ a, b ] : [a] );
      return this._context.createImageData( a, b );
    },
    
    createImageDataHD: function( a, b ) {
      command( 'createImageDataHD', [ a, b ] );
      return this._context.createImageDataHD( a, b );
    },
    
    getImageData: function( sx, sy, sw, sh ) {
      command( 'getImageData', [ sx, sy, sw, sh ] );
      return this._context.getImageData( sx, sy, sw, sh );
    },
    
    getImageDataHD: function( sx, sy, sw, sh ) {
      command( 'getImageDataHD', [ sx, sy, sw, sh ] );
      return this._context.getImageDataHD( sx, sy, sw, sh );
    },
    
    putImageData: function( imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight ) {
      command( 'putImageData', dirtyX !== undefined ? [ imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight ] : [ imageData, dx, dy ] );
      this._context.putImageData( imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight );
    },
    
    putImageDataHD: function( imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight ) {
      command( 'putImageDataHD', dirtyX !== undefined ? [ imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight ] : [ imageData, dx, dy ] );
      this._context.putImageDataHD( imageData, dx, dy, dirtyX, dirtyY, dirtyWidth, dirtyHeight );
    },
    
    /*---------------------------------------------------------------------------*
    * CanvasDrawingStyles
    *----------------------------------------------------------------------------*/
    
    get lineWidth() {
      attributeGet( 'lineWidth' );
      return this._context.lineWidth;
    },
    
    set lineWidth( value ) {
      attributeSet( 'lineWidth', value );
      this._context.lineWidth = value;
    },
    
    get lineCap() {
      attributeGet( 'lineCap' );
      return this._context.lineCap;
    },
    
    set lineCap( value ) {
      attributeSet( 'lineCap', value );
      this._context.lineCap = value;
    },
    
    get lineJoin() {
      attributeGet( 'lineJoin' );
      return this._context.lineJoin;
    },
    
    set lineJoin( value ) {
      attributeSet( 'lineJoin', value );
      this._context.lineJoin = value;
    },
    
    get miterLimit() {
      attributeGet( 'miterLimit' );
      return this._context.miterLimit;
    },
    
    set miterLimit( value ) {
      attributeSet( 'miterLimit', value );
      this._context.miterLimit = value;
    },
    
    setLineDash: function( segments ) {
      command( 'setLineDash', [ segments ] );
      this._context.setLineDash( segments );
    },
    
    getLineDash: function() {
      command( 'getLineDash' );
      return this._context.getLineDash();
    },
    
    get lineDashOffset() {
      attributeGet( 'lineDashOffset' );
      return this._context.lineDashOffset;
    },
    
    set lineDashOffset( value ) {
      attributeSet( 'lineDashOffset', value );
      this._context.lineDashOffset = value;
    },
    
    get font() {
      attributeGet( 'font' );
      return this._context.font;
    },
    
    set font( value ) {
      attributeSet( 'font', value );
      this._context.font = value;
    },
    
    get textAlign() {
      attributeGet( 'textAlign' );
      return this._context.textAlign;
    },
    
    set textAlign( value ) {
      attributeSet( 'textAlign', value );
      this._context.textAlign = value;
    },
    
    get textBaseline() {
      attributeGet( 'textBaseline' );
      return this._context.textBaseline;
    },
    
    set textBaseline( value ) {
      attributeSet( 'textBaseline', value );
      this._context.textBaseline = value;
    },
    
    get direction() {
      attributeGet( 'direction' );
      return this._context.direction;
    },
    
    set direction( value ) {
      attributeSet( 'direction', value );
      this._context.direction = value;
    },
    
    /*---------------------------------------------------------------------------*
    * CanvasPathMethods
    *----------------------------------------------------------------------------*/
    
    closePath: function() {
      command( 'closePath' );
      this._context.closePath();
    },
    
    moveTo: function( x, y ) {
      command( 'moveTo', [ x, y ] );
      this._context.moveTo( x, y );
    },
    
    lineTo: function( x, y ) {
      command( 'lineTo', [ x, y ] );
      this._context.lineTo( x, y );
    },
    
    quadraticCurveTo: function( cpx, cpy, x, y ) {
      command( 'quadraticCurveTo', [ cpx, cpy, x, y ] );
      this._context.quadraticCurveTo( cpx, cpy, x, y );
    },
    
    bezierCurveTo: function( cp1x, cp1y, cp2x, cp2y, x, y ) {
      command( 'bezierCurveTo', [ cp1x, cp1y, cp2x, cp2y, x, y ] );
      this._context.bezierCurveTo( cp1x, cp1y, cp2x, cp2y, x, y );
    },
    
    arcTo: function( x1, y1, x2, y2, radiusX, radiusY, rotation ) {
      command( 'arcTo', radiusY !== undefined ? [ x1, y1, x2, y2, radiusX, radiusY, rotation ] : [ x1, y1, x2, y2, radiusX ] );
      this._context.arcTo( x1, y1, x2, y2, radiusX, radiusY, rotation );
    },
    
    rect: function( x, y, w, h ) {
      command( 'rect', [ x, y, w, h ] );
      this._context.rect( x, y, w, h );
    },
    
    arc: function( x, y, radius, startAngle, endAngle, anticlockwise ) {
      command( 'arc', [ x, y, radius, startAngle, endAngle, anticlockwise ] );
      this._context.arc( x, y, radius, startAngle, endAngle, anticlockwise );
    },
    
    ellipse: function( x, y, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) {
      command( 'ellipse', [ x, y, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ] );
      this._context.ellipse( x, y, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise );
    }
  };
  
  return DebugContext;
} );



// Copyright 2002-2012, University of Colorado

/*
 * An event in Scenery that has similar event-handling characteristics to DOM events.
 * The original DOM event (if any) is available as event.domEvent.
 *
 * Multiple events can be triggered by a single domEvent, so don't assume it is unique.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */
 
define('SCENERY/input/Event',['require','ASSERT/assert','SCENERY/scenery'], function( require ) {
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Event = function( args ) {
    // ensure that all of the required args are supplied
    assert && assert( args.trail &&
                      args.type &&
                      args.pointer &&
                      args.domEvent &&
                      args.target, 'Missing required scenery.Event argument' );
    
    this.handled = false;
    this.aborted = false;
    
    // {Trail} path to the leaf-most node, ordered list, from root to leaf
    this.trail = args.trail;
    
    // {String} what event was triggered on the listener
    this.type = args.type;
    
    // {Pointer}
    this.pointer = args.pointer;
    
    // raw DOM InputEvent (TouchEvent, PointerEvent, MouseEvent,...)
    this.domEvent = args.domEvent;
    
    // {Node} whatever node you attached the listener to, or null when firing events on a Pointer
    this.currentTarget = args.currentTarget;
    
    // {Node} leaf-most node in trail
    this.target = args.trail;
    
    // TODO: add extended information based on an event here?
  };
  var Event = scenery.Event;
  
  Event.prototype = {
    constructor: Event,
    
    // like DOM Event.stopPropagation(), but named differently to indicate it doesn't fire that behavior on the underlying DOM event
    handle: function() {
      this.handled = true;
    },
    
    // like DOM Event.stopImmediatePropagation(), but named differently to indicate it doesn't fire that behavior on the underlying DOM event
    abort: function() {
      this.handled = true;
      this.aborted = true;
    }
  };
  
  return Event;
} );


define('DOT/dot',['require'], function( require ) {
  // will be filled in by other modules
  return {};
} );

// Copyright 2002-2012, University of Colorado

/**
 * Utility functions for Dot, placed into the dot.X namespace.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Util',['require','ASSERT/assert','DOT/dot'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'dot' );
  
  var dot = require( 'DOT/dot' );
  // require( 'DOT/Vector2' ); // Require.js doesn't like the circular reference
  
  dot.Util = {
    testAssert: function() {
      return 'assert.dot: ' + ( assert ? 'true' : 'false' );
    },
    
    clamp: function( value, min, max ) {
      if ( value < min ) {
        return min;
      }
      else if ( value > max ) {
        return max;
      }
      else {
        return value;
      }
    },
    
    // Returns an array of integers from A to B (including both A to B)
    rangeInclusive: function( a, b ) {
      if ( b < a ) {
        return [];
      }
      var result = new Array( b - a + 1 );
      for ( var i = a; i <= b; i++ ) {
        result[i-a] = i;
      }
      return result;
    },
    
    // Returns an array of integers between A and B (excluding both A to B)
    rangeExclusive: function( a, b ) {
      return Util.rangeInclusive( a + 1, b - 1 );
    },
    
    toRadians: function( degrees ) {
      return Math.PI * degrees / 180;
    },
    
    toDegrees: function( radians ) {
      return 180 * radians / Math.PI;
    },
    
    // intersection between the line from p1-p2 and the line from p3-p4
    lineLineIntersection: function( p1, p2, p3, p4 ) {
      return new dot.Vector2(
        ( ( p1.x * p2.y - p1.y * p2.x ) * ( p3.x - p4.x ) - ( p1.x - p2.x ) * ( p3.x * p4.y - p3.y * p4.x ) ) / ( ( p1.x - p2.x ) * ( p3.y - p4.y ) - ( p1.y - p2.y ) * ( p3.x - p4.x ) ),
        ( ( p1.x * p2.y - p1.y * p2.x ) * ( p3.y - p4.y ) - ( p1.y - p2.y ) * ( p3.x * p4.y - p3.y * p4.x ) ) / ( ( p1.x - p2.x ) * ( p3.y - p4.y ) - ( p1.y - p2.y ) * ( p3.x - p4.x ) )
      );
    },
    
    // return an array of real roots of ax^2 + bx + c = 0
    solveQuadraticRootsReal: function( a, b, c ) {
      var discriminant = b * b - 4 * a * c;
      if ( discriminant < 0 ) {
        return [];
      }
      var sqrt = Math.sqrt( discriminant );
      // TODO: how to handle if discriminant is 0? give unique root or double it?
      // TODO: probably just use Complex for the future
      return [
        ( -b - sqrt ) / ( 2 * a ),
        ( -b + sqrt ) / ( 2 * a )
      ];
    },
    
    // return an array of real roots of ax^3 + bx^2 + cx + d = 0
    solveCubicRootsReal: function( a, b, c, d ) {
      // TODO: a Complex type!
      if ( a === 0 ) {
        return Util.solveQuadraticRootsReal( b, c, d );
      }
      if ( d === 0 ) {
        return Util.solveQuadraticRootsReal( a, b, c );
      }
      
      b /= a;
      c /= a;
      d /= a;
      
      var s, t;
      var q = ( 3.0 * c - ( b * b ) ) / 9;
      var r = ( -(27 * d) + b * (9 * c - 2 * (b * b)) ) / 54;
      var discriminant = q  * q  * q + r  * r;
      var b3 = b / 3;
      
      if ( discriminant > 0 ) {
        // a single real root
        var dsqrt = Math.sqrt( discriminant );
        return [ Util.cubeRoot( r + dsqrt ) + Util.cubeRoot( r - dsqrt ) - b3 ];
      }
      
      // three real roots
      if ( discriminant === 0 ) {
        // contains a double root
        var rsqrt = Util.cubeRoot( r );
        var doubleRoot = b3 - rsqrt;
        return [ -b3 + 2 * rsqrt, doubleRoot, doubleRoot ];
      } else {
        // all unique
        var qX = -q * q * q;
        qX = Math.acos( r / Math.sqrt( qX ) );
        var rr = 2 * Math.sqrt( -q );
        return [
          -b3 + rr * Math.cos( qX / 3 ),
          -b3 + rr * Math.cos( ( qX + 2 * Math.PI ) / 3 ),
          -b3 + rr * Math.cos( ( qX + 4 * Math.PI ) / 3 )
        ];
      }
    },
    
    cubeRoot: function( x ) {
      return x >= 0 ? Math.pow( x, 1/3 ) : -Math.pow( -x, 1/3 );
    },

    //Linearly interpolate two points and evaluate the line equation for a third point
    //Arguments are in the form x1=>y1, x2=>y2, x3=> ???
    linear: function( x1, y1, x2, y2, x3 ) {
      return (y2 - y1) / (x2 - x1) * (x3 - x1 ) + y1;
    }
  };
  var Util = dot.Util;
  
  // make these available in the main namespace directly (for now)
  dot.testAssert = Util.testAssert;
  dot.clamp = Util.clamp;
  dot.rangeInclusive = Util.rangeInclusive;
  dot.rangeExclusive = Util.rangeExclusive;
  dot.toRadians = Util.toRadians;
  dot.toDegrees = Util.toDegrees;
  
  return Util;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Basic 2-dimensional vector
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Vector2',['require','ASSERT/assert','DOT/dot','DOT/Util'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'dot' );
  
  var dot = require( 'DOT/dot' );
  
  require( 'DOT/Util' );
  // require( 'DOT/Vector3' ); // commented out since Require.js complains about the circular dependency
  
  dot.Vector2 = function( x, y ) {
    // allow optional parameters
    this.x = x || 0;
    this.y = y || 0;
  };
  var Vector2 = dot.Vector2;
  
  Vector2.createPolar = function( magnitude, angle ) {
    return new Vector2( Math.cos( angle ), Math.sin( angle ) ).timesScalar( magnitude );
  };
  
  Vector2.prototype = {
    constructor: Vector2,
    
    isVector2: true,
    
    dimension: 2,
    
    magnitude: function() {
      return Math.sqrt( this.magnitudeSquared() );
    },
    
    magnitudeSquared: function() {
      return this.dot( this );
    },
    
    // the distance between this vector (treated as a point) and another point
    distance: function( point ) {
      return this.minus( point ).magnitude();
    },
    
    // the squared distance between this vector (treated as a point) and another point
    distanceSquared: function( point ) {
      return this.minus( point ).magnitudeSquared();
    },
    
    dot: function( v ) {
      return this.x * v.x + this.y * v.y;
    },
    
    equals: function( other ) {
      return this.x === other.x && this.y === other.y;
    },
    
    equalsEpsilon: function( other, epsilon ) {
      if ( !epsilon ) {
        epsilon = 0;
      }
      return Math.abs( this.x - other.x ) + Math.abs( this.y - other.y ) <= epsilon;
    },
    
    isFinite: function() {
      return isFinite( this.x ) && isFinite( this.y );
    },
    
    /*---------------------------------------------------------------------------*
     * Immutables
     *----------------------------------------------------------------------------*/
    
    copy: function() {
      return new Vector2( this.x, this.y );
    },
    
    // z component of the equivalent 3-dimensional cross product (this.x, this.y,0) x (v.x, v.y, 0)
    crossScalar: function( v ) {
      return this.x * v.y  - this.y * v.x;
    },
    
    normalized: function() {
      var mag = this.magnitude();
      if ( mag === 0 ) {
        throw new Error( "Cannot normalize a zero-magnitude vector" );
      }
      else {
        return new Vector2( this.x / mag, this.y / mag );
      }
    },
    
    timesScalar: function( scalar ) {
      return new Vector2( this.x * scalar, this.y * scalar );
    },
    
    times: function( scalar ) {
      // make sure it's not a vector!
      assert && assert( scalar.dimension === undefined );
      return this.timesScalar( scalar );
    },
    
    componentTimes: function( v ) {
      return new Vector2( this.x * v.x, this.y * v.y );
    },
    
    plus: function( v ) {
      return new Vector2( this.x + v.x, this.y + v.y );
    },
    
    plusScalar: function( scalar ) {
      return new Vector2( this.x + scalar, this.y + scalar );
    },
    
    minus: function( v ) {
      return new Vector2( this.x - v.x, this.y - v.y );
    },
    
    minusScalar: function( scalar ) {
      return new Vector2( this.x - scalar, this.y - scalar );
    },
    
    dividedScalar: function( scalar ) {
      return new Vector2( this.x / scalar, this.y / scalar );
    },
    
    negated: function() {
      return new Vector2( -this.x, -this.y );
    },
    
    angle: function() {
      return Math.atan2( this.y, this.x );
    },
    
    // equivalent to a -PI/2 rotation (right hand rotation)
    perpendicular: function() {
      return new Vector2( this.y, -this.x );
    },
    
    angleBetween: function( v ) {
      return Math.acos( dot.clamp( this.normalized().dot( v.normalized() ), -1, 1 ) );
    },
    
    rotated: function( angle ) {
      return Vector2.createPolar( this.magnitude(), this.angle() + angle );
    },
      
    // linear interpolation from this (ratio=0) to vector (ratio=1)
    blend: function( vector, ratio ) {
      return this.plus( vector.minus( this ).times( ratio ) );
    },
    
    toString: function() {
      return "Vector2(" + this.x + ", " + this.y + ")";
    },
    
    toVector3: function() {
      return new dot.Vector3( this.x, this.y );
    },
    
    /*---------------------------------------------------------------------------*
     * Mutables
     *----------------------------------------------------------------------------*/
     
    set: function( x, y ) {
      this.x = x;
      this.y = y;
      return this;
    },
    
    setX: function( x ) {
      this.x = x;
      return this;
    },
    
    setY: function( y ) {
      this.y = y;
      return this;
    },
    
    add: function( v ) {
      this.x += v.x;
      this.y += v.y;
      return this;
    },
    
    addScalar: function( scalar ) {
      this.x += scalar;
      this.y += scalar;
      return this;
    },
    
    subtract: function( v ) {
      this.x -= v.x;
      this.y -= v.y;
      return this;
    },
    
    subtractScalar: function( scalar ) {
      this.x -= scalar;
      this.y -= scalar;
      return this;
    },
    
    componentMultiply: function( v ) {
      this.x *= v.x;
      this.y *= v.y;
      return this;
    },
    
    divideScalar: function( scalar ) {
      this.x /= scalar;
      this.y /= scalar;
      return this;
    },
    
    negate: function() {
      this.x = -this.x;
      this.y = -this.y;
      return this;
    }
    
  };
  
  /*---------------------------------------------------------------------------*
   * Immutable Vector form
   *----------------------------------------------------------------------------*/
  Vector2.Immutable = function( x, y ) {
    this.x = x || 0;
    this.y = y || 0;
  };
  var Immutable = Vector2.Immutable;
  
  Immutable.prototype = new Vector2();
  Immutable.prototype.constructor = Immutable;
  
  // throw errors whenever a mutable method is called on our immutable vector
  Immutable.mutableOverrideHelper = function( mutableFunctionName ) {
    Immutable.prototype[mutableFunctionName] = function() {
      throw new Error( "Cannot call mutable method '" + mutableFunctionName + "' on immutable Vector2" );
    };
  };
  
  // TODO: better way to handle this list?
  Immutable.mutableOverrideHelper( 'set' );
  Immutable.mutableOverrideHelper( 'setX' );
  Immutable.mutableOverrideHelper( 'setY' );
  Immutable.mutableOverrideHelper( 'copy' );
  Immutable.mutableOverrideHelper( 'add' );
  Immutable.mutableOverrideHelper( 'addScalar' );
  Immutable.mutableOverrideHelper( 'subtract' );
  Immutable.mutableOverrideHelper( 'subtractScalar' );
  Immutable.mutableOverrideHelper( 'componentMultiply' );
  Immutable.mutableOverrideHelper( 'divideScalar' );
  Immutable.mutableOverrideHelper( 'negate' );
  
  // helpful immutable constants
  Vector2.ZERO = new Immutable( 0, 0 );
  Vector2.X_UNIT = new Immutable( 1, 0 );
  Vector2.Y_UNIT = new Immutable( 0, 1 );
  
  return Vector2;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Basic 4-dimensional vector
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Vector4',['require','ASSERT/assert','DOT/dot','DOT/Util'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'dot' );
  
  var dot = require( 'DOT/dot' );
  
  require( 'DOT/Util' );
  // require( 'DOT/Vector3' ); // commented out so Require.js doesn't complain about the circular dependency
  
  dot.Vector4 = function( x, y, z, w ) {
    // allow optional parameters
    this.x = x || 0;
    this.y = y || 0;
    this.z = z || 0;
    this.w = w !== undefined ? w : 1; // since w could be zero!
  };
  var Vector4 = dot.Vector4;
  
  Vector4.prototype = {
    constructor: Vector4,

    magnitude: function() {
      return Math.sqrt( this.magnitudeSquared() );
    },

    magnitudeSquared: function() {
      this.dot( this );
    },
    
    // the distance between this vector (treated as a point) and another point
    distance: function( point ) {
      return this.minus( point ).magnitude();
    },
    
    // the squared distance between this vector (treated as a point) and another point
    distanceSquared: function( point ) {
      return this.minus( point ).magnitudeSquared();
    },

    dot: function( v ) {
      return this.x * v.x + this.y * v.y + this.z * v.z + this.w * v.w;
    },
    
    isFinite: function() {
      return isFinite( this.x ) && isFinite( this.y ) && isFinite( this.z ) && isFinite( this.w );
    },

    /*---------------------------------------------------------------------------*
     * Immutables
     *----------------------------------------------------------------------------*/

    normalized: function() {
      var mag = this.magnitude();
      if ( mag === 0 ) {
        throw new Error( "Cannot normalize a zero-magnitude vector" );
      }
      else {
        return new Vector4( this.x / mag, this.y / mag, this.z / mag, this.w / mag );
      }
    },

    timesScalar: function( scalar ) {
      return new Vector4( this.x * scalar, this.y * scalar, this.z * scalar, this.w * scalar );
    },

    times: function( scalar ) {
      // make sure it's not a vector!
      assert && assert( scalar.dimension === undefined );
      return this.timesScalar( scalar );
    },

    componentTimes: function( v ) {
      return new Vector4( this.x * v.x, this.y * v.y, this.z * v.z, this.w * v.w );
    },

    plus: function( v ) {
      return new Vector4( this.x + v.x, this.y + v.y, this.z + v.z, this.w + v.w );
    },

    plusScalar: function( scalar ) {
      return new Vector4( this.x + scalar, this.y + scalar, this.z + scalar, this.w + scalar );
    },

    minus: function( v ) {
      return new Vector4( this.x - v.x, this.y - v.y, this.z - v.z, this.w - v.w );
    },

    minusScalar: function( scalar ) {
      return new Vector4( this.x - scalar, this.y - scalar, this.z - scalar, this.w - scalar );
    },

    dividedScalar: function( scalar ) {
      return new Vector4( this.x / scalar, this.y / scalar, this.z / scalar, this.w / scalar );
    },

    negated: function() {
      return new Vector4( -this.x, -this.y, -this.z, -this.w );
    },

    angleBetween: function( v ) {
      return Math.acos( dot.clamp( this.normalized().dot( v.normalized() ), -1, 1 ) );
    },
    
    // linear interpolation from this (ratio=0) to vector (ratio=1)
    blend: function( vector, ratio ) {
      return this.plus( vector.minus( this ).times( ratio ) );
    },

    toString: function() {
      return "Vector4(" + this.x + ", " + this.y + ", " + this.z + ", " + this.w + ")";
    },

    toVector3: function() {
      return new dot.Vector3( this.x, this.y, this.z );
    },

    /*---------------------------------------------------------------------------*
     * Mutables
     *----------------------------------------------------------------------------*/

    set: function( x, y, z, w ) {
      this.x = x;
      this.y = y;
      this.z = z;
      this.w = w;
    },

    setX: function( x ) {
      this.x = x;
    },

    setY: function( y ) {
      this.y = y;
    },

    setZ: function( z ) {
      this.z = z;
    },

    setW: function( w ) {
      this.w = w;
    },

    copy: function( v ) {
      this.x = v.x;
      this.y = v.y;
      this.z = v.z;
      this.w = v.w;
    },

    add: function( v ) {
      this.x += v.x;
      this.y += v.y;
      this.z += v.z;
      this.w += v.w;
    },

    addScalar: function( scalar ) {
      this.x += scalar;
      this.y += scalar;
      this.z += scalar;
      this.w += scalar;
    },

    subtract: function( v ) {
      this.x -= v.x;
      this.y -= v.y;
      this.z -= v.z;
      this.w -= v.w;
    },

    subtractScalar: function( scalar ) {
      this.x -= scalar;
      this.y -= scalar;
      this.z -= scalar;
      this.w -= scalar;
    },

    componentMultiply: function( v ) {
      this.x *= v.x;
      this.y *= v.y;
      this.z *= v.z;
      this.w *= v.w;
    },

    divideScalar: function( scalar ) {
      this.x /= scalar;
      this.y /= scalar;
      this.z /= scalar;
      this.w /= scalar;
    },

    negate: function() {
      this.x = -this.x;
      this.y = -this.y;
      this.z = -this.z;
      this.w = -this.w;
    },
    
    equals: function( other ) {
      return this.x === other.x && this.y === other.y && this.z === other.z && this.w === other.w;
    },
    
    equalsEpsilon: function( other, epsilon ) {
      if ( !epsilon ) {
        epsilon = 0;
      }
      return Math.abs( this.x - other.x ) + Math.abs( this.y - other.y ) + Math.abs( this.z - other.z ) + Math.abs( this.w - other.w ) <= epsilon;
    },

    isVector4: true,

    dimension: 4

  };

  /*---------------------------------------------------------------------------*
   * Immutable Vector form
   *----------------------------------------------------------------------------*/
  Vector4.Immutable = function( x, y, z, w ) {
    this.x = x || 0;
    this.y = y || 0;
    this.z = z || 0;
    this.w = w !== undefined ? w : 1;
  };
  var Immutable = Vector4.Immutable;

  Immutable.prototype = new Vector4();
  Immutable.prototype.constructor = Immutable;

  // throw errors whenever a mutable method is called on our immutable vector
  Immutable.mutableOverrideHelper = function( mutableFunctionName ) {
    Immutable.prototype[mutableFunctionName] = function() {
      throw new Error( "Cannot call mutable method '" + mutableFunctionName + "' on immutable Vector4" );
    };
  };

  // TODO: better way to handle this list?
  Immutable.mutableOverrideHelper( 'set' );
  Immutable.mutableOverrideHelper( 'setX' );
  Immutable.mutableOverrideHelper( 'setY' );
  Immutable.mutableOverrideHelper( 'setZ' );
  Immutable.mutableOverrideHelper( 'setW' );
  Immutable.mutableOverrideHelper( 'copy' );
  Immutable.mutableOverrideHelper( 'add' );
  Immutable.mutableOverrideHelper( 'addScalar' );
  Immutable.mutableOverrideHelper( 'subtract' );
  Immutable.mutableOverrideHelper( 'subtractScalar' );
  Immutable.mutableOverrideHelper( 'componentMultiply' );
  Immutable.mutableOverrideHelper( 'divideScalar' );
  Immutable.mutableOverrideHelper( 'negate' );

  // helpful immutable constants
  Vector4.ZERO = new Immutable( 0, 0, 0, 0 );
  Vector4.X_UNIT = new Immutable( 1, 0, 0, 0 );
  Vector4.Y_UNIT = new Immutable( 0, 1, 0, 0 );
  Vector4.Z_UNIT = new Immutable( 0, 0, 1, 0 );
  Vector4.W_UNIT = new Immutable( 0, 0, 0, 1 );
  
  return Vector4;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Basic 3-dimensional vector
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Vector3',['require','ASSERT/assert','DOT/dot','DOT/Util','DOT/Vector2','DOT/Vector4'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'dot' );
  
  var dot = require( 'DOT/dot' );
  
  require( 'DOT/Util' );
  require( 'DOT/Vector2' );
  require( 'DOT/Vector4' );

  dot.Vector3 = function( x, y, z ) {
    // allow optional parameters
    this.x = x || 0;
    this.y = y || 0;
    this.z = z || 0;
  };
  var Vector3 = dot.Vector3;

  Vector3.prototype = {
    constructor: Vector3,

    magnitude: function() {
      return Math.sqrt( this.magnitudeSquared() );
    },

    magnitudeSquared: function() {
      return this.dot( this );
    },
    
    // the distance between this vector (treated as a point) and another point
    distance: function( point ) {
      return this.minus( point ).magnitude();
    },
    
    // the squared distance between this vector (treated as a point) and another point
    distanceSquared: function( point ) {
      return this.minus( point ).magnitudeSquared();
    },

    dot: function( v ) {
      return this.x * v.x + this.y * v.y + this.z * v.z;
    },
    
    isFinite: function() {
      return isFinite( this.x ) && isFinite( this.y ) && isFinite( this.z );
    },

    /*---------------------------------------------------------------------------*
     * Immutables
     *----------------------------------------------------------------------------*/

    cross: function( v ) {
      return new Vector3(
          this.y * v.z - this.z * v.y,
          this.z * v.x - this.x * v.z,
          this.x * v.y - this.y * v.x
      );
    },

    normalized: function() {
      var mag = this.magnitude();
      if ( mag === 0 ) {
        throw new Error( "Cannot normalize a zero-magnitude vector" );
      }
      else {
        return new Vector3( this.x / mag, this.y / mag, this.z / mag );
      }
    },

    timesScalar: function( scalar ) {
      return new Vector3( this.x * scalar, this.y * scalar, this.z * scalar );
    },

    times: function( scalar ) {
      // make sure it's not a vector!
      assert && assert( scalar.dimension === undefined );
      return this.timesScalar( scalar );
    },

    componentTimes: function( v ) {
      return new Vector3( this.x * v.x, this.y * v.y, this.z * v.z );
    },

    plus: function( v ) {
      return new Vector3( this.x + v.x, this.y + v.y, this.z + v.z );
    },

    plusScalar: function( scalar ) {
      return new Vector3( this.x + scalar, this.y + scalar, this.z + scalar );
    },

    minus: function( v ) {
      return new Vector3( this.x - v.x, this.y - v.y, this.z - v.z );
    },

    minusScalar: function( scalar ) {
      return new Vector3( this.x - scalar, this.y - scalar, this.z - scalar );
    },

    dividedScalar: function( scalar ) {
      return new Vector3( this.x / scalar, this.y / scalar, this.z / scalar );
    },

    negated: function() {
      return new Vector3( -this.x, -this.y, -this.z );
    },

    angleBetween: function( v ) {
      return Math.acos( dot.clamp( this.normalized().dot( v.normalized() ), -1, 1 ) );
    },
    
    // linear interpolation from this (ratio=0) to vector (ratio=1)
    blend: function( vector, ratio ) {
      return this.plus( vector.minus( this ).times( ratio ) );
    },

    toString: function() {
      return "Vector3(" + this.x + ", " + this.y + ", " + this.z + ")";
    },

    toVector2: function() {
      return new dot.Vector2( this.x, this.y );
    },

    toVector4: function() {
      return new dot.Vector4( this.x, this.y, this.z );
    },

    /*---------------------------------------------------------------------------*
     * Mutables
     *----------------------------------------------------------------------------*/

    set: function( x, y, z ) {
      this.x = x;
      this.y = y;
      this.z = z;
    },

    setX: function( x ) {
      this.x = x;
    },

    setY: function( y ) {
      this.y = y;
    },

    setZ: function( z ) {
      this.z = z;
    },

    copy: function( v ) {
      this.x = v.x;
      this.y = v.y;
      this.z = v.z;
    },

    add: function( v ) {
      this.x += v.x;
      this.y += v.y;
      this.z += v.z;
    },

    addScalar: function( scalar ) {
      this.x += scalar;
      this.y += scalar;
      this.z += scalar;
    },

    subtract: function( v ) {
      this.x -= v.x;
      this.y -= v.y;
      this.z -= v.z;
    },

    subtractScalar: function( scalar ) {
      this.x -= scalar;
      this.y -= scalar;
      this.z -= scalar;
    },

    componentMultiply: function( v ) {
      this.x *= v.x;
      this.y *= v.y;
      this.z *= v.z;
    },

    divideScalar: function( scalar ) {
      this.x /= scalar;
      this.y /= scalar;
      this.z /= scalar;
    },

    negate: function() {
      this.x = -this.x;
      this.y = -this.y;
      this.z = -this.z;
    },
    
    equals: function( other ) {
      return this.x === other.x && this.y === other.y && this.z === other.z;
    },
    
    equalsEpsilon: function( other, epsilon ) {
      if ( !epsilon ) {
        epsilon = 0;
      }
      return Math.abs( this.x - other.x ) + Math.abs( this.y - other.y ) + Math.abs( this.z - other.z ) <= epsilon;
    },

    isVector3: true,

    dimension: 3

  };

  /*---------------------------------------------------------------------------*
   * Immutable Vector form
   *----------------------------------------------------------------------------*/
  Vector3.Immutable = function( x, y, z ) {
    this.x = x || 0;
    this.y = y || 0;
    this.z = z || 0;
  };
  var Immutable = Vector3.Immutable;

  Immutable.prototype = new Vector3();
  Immutable.prototype.constructor = Immutable;

  // throw errors whenever a mutable method is called on our immutable vector
  Immutable.mutableOverrideHelper = function( mutableFunctionName ) {
    Immutable.prototype[mutableFunctionName] = function() {
      throw new Error( "Cannot call mutable method '" + mutableFunctionName + "' on immutable Vector3" );
    };
  };

  // TODO: better way to handle this list?
  Immutable.mutableOverrideHelper( 'set' );
  Immutable.mutableOverrideHelper( 'setX' );
  Immutable.mutableOverrideHelper( 'setY' );
  Immutable.mutableOverrideHelper( 'setZ' );
  Immutable.mutableOverrideHelper( 'copy' );
  Immutable.mutableOverrideHelper( 'add' );
  Immutable.mutableOverrideHelper( 'addScalar' );
  Immutable.mutableOverrideHelper( 'subtract' );
  Immutable.mutableOverrideHelper( 'subtractScalar' );
  Immutable.mutableOverrideHelper( 'componentMultiply' );
  Immutable.mutableOverrideHelper( 'divideScalar' );
  Immutable.mutableOverrideHelper( 'negate' );

  // helpful immutable constants
  Vector3.ZERO = new Immutable( 0, 0, 0 );
  Vector3.X_UNIT = new Immutable( 1, 0, 0 );
  Vector3.Y_UNIT = new Immutable( 0, 1, 0 );
  Vector3.Z_UNIT = new Immutable( 0, 0, 1 );
  
  return Vector3;
} );

// Copyright 2002-2012, University of Colorado

/**
 * 4-dimensional Matrix
 *
 * TODO: consider adding affine flag if it will help performance (a la Matrix3)
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Matrix4',['require','DOT/dot','DOT/Vector3','DOT/Vector4'], function( require ) {
  
  
  var dot = require( 'DOT/dot' );
  
  require( 'DOT/Vector3' );
  require( 'DOT/Vector4' );
  
  var Float32Array = window.Float32Array || Array;
  
  dot.Matrix4 = function( v00, v01, v02, v03, v10, v11, v12, v13, v20, v21, v22, v23, v30, v31, v32, v33, type ) {

    // entries stored in column-major format
    this.entries = new Float32Array( 16 );

    this.rowMajor( v00 === undefined ? 1 : v00, v01 || 0, v02 || 0, v03 || 0,
             v10 || 0, v11 === undefined ? 1 : v11, v12 || 0, v13 || 0,
             v20 || 0, v21 || 0, v22 === undefined ? 1 : v22, v23 || 0,
             v30 || 0, v31 || 0, v32 || 0, v33 === undefined ? 1 : v33,
             type );
  };
  var Matrix4 = dot.Matrix4;

  Matrix4.Types = {
    OTHER: 0, // default
    IDENTITY: 1,
    TRANSLATION_3D: 2,
    SCALING: 3

    // TODO: possibly add rotations
  };

  var Types = Matrix4.Types;

  Matrix4.identity = function() {
    return new Matrix4( 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1,
              Types.IDENTITY );
  };

  Matrix4.translation = function( x, y, z ) {
    return new Matrix4( 1, 0, 0, x,
              0, 1, 0, y,
              0, 0, 1, z,
              0, 0, 0, 1,
              Types.TRANSLATION_3D );
  };

  Matrix4.translationFromVector = function( v ) { return Matrix4.translation( v.x, v.y, v.z ); };

  Matrix4.scaling = function( x, y, z ) {
    // allow using one parameter to scale everything
    y = y === undefined ? x : y;
    z = z === undefined ? x : z;

    return new Matrix4( x, 0, 0, 0,
              0, y, 0, 0,
              0, 0, z, 0,
              0, 0, 0, 1,
              Types.SCALING );
  };

  // axis is a normalized Vector3, angle in radians.
  Matrix4.rotationAxisAngle = function( axis, angle ) {
    var c = Math.cos( angle );
    var s = Math.sin( angle );
    var C = 1 - c;

    return new Matrix4( axis.x * axis.x * C + c, axis.x * axis.y * C - axis.z * s, axis.x * axis.z * C + axis.y * s, 0,
              axis.y * axis.x * C + axis.z * s, axis.y * axis.y * C + c, axis.y * axis.z * C - axis.x * s, 0,
              axis.z * axis.x * C - axis.y * s, axis.z * axis.y * C + axis.x * s, axis.z * axis.z * C + c, 0,
              0, 0, 0, 1,
              Types.OTHER );
  };

  // TODO: add in rotation from quaternion, and from quat + translation

  Matrix4.rotationX = function( angle ) {
    var c = Math.cos( angle );
    var s = Math.sin( angle );

    return new Matrix4( 1, 0, 0, 0,
              0, c, -s, 0,
              0, s, c, 0,
              0, 0, 0, 1,
              Types.OTHER );
  };

  Matrix4.rotationY = function( angle ) {
    var c = Math.cos( angle );
    var s = Math.sin( angle );

    return new Matrix4( c, 0, s, 0,
              0, 1, 0, 0,
              -s, 0, c, 0,
              0, 0, 0, 1,
              Types.OTHER );
  };

  Matrix4.rotationZ = function( angle ) {
    var c = Math.cos( angle );
    var s = Math.sin( angle );

    return new Matrix4( c, -s, 0, 0,
              s, c, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1,
              Types.OTHER );
  };

  // aspect === width / height
  Matrix4.gluPerspective = function( fovYRadians, aspect, zNear, zFar ) {
    var cotangent = Math.cos( fovYRadians ) / Math.sin( fovYRadians );

    return new Matrix4( cotangent / aspect, 0, 0, 0,
              0, cotangent, 0, 0,
              0, 0, ( zFar + zNear ) / ( zNear - zFar ), ( 2 * zFar * zNear ) / ( zNear - zFar ),
              0, 0, -1, 0 );
  };

  Matrix4.prototype = {
    constructor: Matrix4,

    rowMajor: function( v00, v01, v02, v03, v10, v11, v12, v13, v20, v21, v22, v23, v30, v31, v32, v33, type ) {
      this.entries[0] = v00;
      this.entries[1] = v10;
      this.entries[2] = v20;
      this.entries[3] = v30;
      this.entries[4] = v01;
      this.entries[5] = v11;
      this.entries[6] = v21;
      this.entries[7] = v31;
      this.entries[8] = v02;
      this.entries[9] = v12;
      this.entries[10] = v22;
      this.entries[11] = v32;
      this.entries[12] = v03;
      this.entries[13] = v13;
      this.entries[14] = v23;
      this.entries[15] = v33;
      this.type = type === undefined ? Types.OTHER : type;
    },

    columnMajor: function( v00, v10, v20, v30, v01, v11, v21, v31, v02, v12, v22, v32, v03, v13, v23, v33, type ) {
      this.rowMajor( v00, v01, v02, v03, v10, v11, v12, v13, v20, v21, v22, v23, v30, v31, v32, v33, type );
    },

    // convenience getters. inline usages of these when performance is critical? TODO: test performance of inlining these, with / without closure compiler
    m00: function() { return this.entries[0]; },
    m01: function() { return this.entries[4]; },
    m02: function() { return this.entries[8]; },
    m03: function() { return this.entries[12]; },
    m10: function() { return this.entries[1]; },
    m11: function() { return this.entries[5]; },
    m12: function() { return this.entries[9]; },
    m13: function() { return this.entries[13]; },
    m20: function() { return this.entries[2]; },
    m21: function() { return this.entries[6]; },
    m22: function() { return this.entries[10]; },
    m23: function() { return this.entries[14]; },
    m30: function() { return this.entries[3]; },
    m31: function() { return this.entries[7]; },
    m32: function() { return this.entries[11]; },
    m33: function() { return this.entries[15]; },

    plus: function( m ) {
      return new Matrix4(
          this.m00() + m.m00(), this.m01() + m.m01(), this.m02() + m.m02(), this.m03() + m.m03(),
          this.m10() + m.m10(), this.m11() + m.m11(), this.m12() + m.m12(), this.m13() + m.m13(),
          this.m20() + m.m20(), this.m21() + m.m21(), this.m22() + m.m22(), this.m23() + m.m23(),
          this.m30() + m.m30(), this.m31() + m.m31(), this.m32() + m.m32(), this.m33() + m.m33()
      );
    },

    minus: function( m ) {
      return new Matrix4(
          this.m00() - m.m00(), this.m01() - m.m01(), this.m02() - m.m02(), this.m03() - m.m03(),
          this.m10() - m.m10(), this.m11() - m.m11(), this.m12() - m.m12(), this.m13() - m.m13(),
          this.m20() - m.m20(), this.m21() - m.m21(), this.m22() - m.m22(), this.m23() - m.m23(),
          this.m30() - m.m30(), this.m31() - m.m31(), this.m32() - m.m32(), this.m33() - m.m33()
      );
    },

    transposed: function() {
      return new Matrix4( this.m00(), this.m10(), this.m20(), this.m30(),
                this.m01(), this.m11(), this.m21(), this.m31(),
                this.m02(), this.m12(), this.m22(), this.m32(),
                this.m03(), this.m13(), this.m23(), this.m33() );
    },

    negated: function() {
      return new Matrix4( -this.m00(), -this.m01(), -this.m02(), -this.m03(),
                -this.m10(), -this.m11(), -this.m12(), -this.m13(),
                -this.m20(), -this.m21(), -this.m22(), -this.m23(),
                -this.m30(), -this.m31(), -this.m32(), -this.m33() );
    },

    inverted: function() {
      // TODO: optimizations for matrix types (like identity)

      var det = this.determinant();

      if ( det !== 0 ) {
        return new Matrix4(
            ( -this.m31() * this.m22() * this.m13() + this.m21() * this.m32() * this.m13() + this.m31() * this.m12() * this.m23() - this.m11() * this.m32() * this.m23() - this.m21() * this.m12() * this.m33() + this.m11() * this.m22() * this.m33() ) / det,
            ( this.m31() * this.m22() * this.m03() - this.m21() * this.m32() * this.m03() - this.m31() * this.m02() * this.m23() + this.m01() * this.m32() * this.m23() + this.m21() * this.m02() * this.m33() - this.m01() * this.m22() * this.m33() ) / det,
            ( -this.m31() * this.m12() * this.m03() + this.m11() * this.m32() * this.m03() + this.m31() * this.m02() * this.m13() - this.m01() * this.m32() * this.m13() - this.m11() * this.m02() * this.m33() + this.m01() * this.m12() * this.m33() ) / det,
            ( this.m21() * this.m12() * this.m03() - this.m11() * this.m22() * this.m03() - this.m21() * this.m02() * this.m13() + this.m01() * this.m22() * this.m13() + this.m11() * this.m02() * this.m23() - this.m01() * this.m12() * this.m23() ) / det,
            ( this.m30() * this.m22() * this.m13() - this.m20() * this.m32() * this.m13() - this.m30() * this.m12() * this.m23() + this.m10() * this.m32() * this.m23() + this.m20() * this.m12() * this.m33() - this.m10() * this.m22() * this.m33() ) / det,
            ( -this.m30() * this.m22() * this.m03() + this.m20() * this.m32() * this.m03() + this.m30() * this.m02() * this.m23() - this.m00() * this.m32() * this.m23() - this.m20() * this.m02() * this.m33() + this.m00() * this.m22() * this.m33() ) / det,
            ( this.m30() * this.m12() * this.m03() - this.m10() * this.m32() * this.m03() - this.m30() * this.m02() * this.m13() + this.m00() * this.m32() * this.m13() + this.m10() * this.m02() * this.m33() - this.m00() * this.m12() * this.m33() ) / det,
            ( -this.m20() * this.m12() * this.m03() + this.m10() * this.m22() * this.m03() + this.m20() * this.m02() * this.m13() - this.m00() * this.m22() * this.m13() - this.m10() * this.m02() * this.m23() + this.m00() * this.m12() * this.m23() ) / det,
            ( -this.m30() * this.m21() * this.m13() + this.m20() * this.m31() * this.m13() + this.m30() * this.m11() * this.m23() - this.m10() * this.m31() * this.m23() - this.m20() * this.m11() * this.m33() + this.m10() * this.m21() * this.m33() ) / det,
            ( this.m30() * this.m21() * this.m03() - this.m20() * this.m31() * this.m03() - this.m30() * this.m01() * this.m23() + this.m00() * this.m31() * this.m23() + this.m20() * this.m01() * this.m33() - this.m00() * this.m21() * this.m33() ) / det,
            ( -this.m30() * this.m11() * this.m03() + this.m10() * this.m31() * this.m03() + this.m30() * this.m01() * this.m13() - this.m00() * this.m31() * this.m13() - this.m10() * this.m01() * this.m33() + this.m00() * this.m11() * this.m33() ) / det,
            ( this.m20() * this.m11() * this.m03() - this.m10() * this.m21() * this.m03() - this.m20() * this.m01() * this.m13() + this.m00() * this.m21() * this.m13() + this.m10() * this.m01() * this.m23() - this.m00() * this.m11() * this.m23() ) / det,
            ( this.m30() * this.m21() * this.m12() - this.m20() * this.m31() * this.m12() - this.m30() * this.m11() * this.m22() + this.m10() * this.m31() * this.m22() + this.m20() * this.m11() * this.m32() - this.m10() * this.m21() * this.m32() ) / det,
            ( -this.m30() * this.m21() * this.m02() + this.m20() * this.m31() * this.m02() + this.m30() * this.m01() * this.m22() - this.m00() * this.m31() * this.m22() - this.m20() * this.m01() * this.m32() + this.m00() * this.m21() * this.m32() ) / det,
            ( this.m30() * this.m11() * this.m02() - this.m10() * this.m31() * this.m02() - this.m30() * this.m01() * this.m12() + this.m00() * this.m31() * this.m12() + this.m10() * this.m01() * this.m32() - this.m00() * this.m11() * this.m32() ) / det,
            ( -this.m20() * this.m11() * this.m02() + this.m10() * this.m21() * this.m02() + this.m20() * this.m01() * this.m12() - this.m00() * this.m21() * this.m12() - this.m10() * this.m01() * this.m22() + this.m00() * this.m11() * this.m22() ) / det
        );
      }
      else {
        throw new Error( "Matrix could not be inverted, determinant === 0" );
      }
    },

    timesMatrix: function( m ) {
      var newType = Types.OTHER;
      if ( this.type === Types.TRANSLATION_3D && m.type === Types.TRANSLATION_3D ) {
        newType = Types.TRANSLATION_3D;
      }
      if ( this.type === Types.SCALING && m.type === Types.SCALING ) {
        newType = Types.SCALING;
      }
      if ( this.type === Types.IDENTITY ) {
        newType = m.type;
      }
      if ( m.type === Types.IDENTITY ) {
        newType = this.type;
      }
      return new Matrix4( this.m00() * m.m00() + this.m01() * m.m10() + this.m02() * m.m20() + this.m03() * m.m30(),
                this.m00() * m.m01() + this.m01() * m.m11() + this.m02() * m.m21() + this.m03() * m.m31(),
                this.m00() * m.m02() + this.m01() * m.m12() + this.m02() * m.m22() + this.m03() * m.m32(),
                this.m00() * m.m03() + this.m01() * m.m13() + this.m02() * m.m23() + this.m03() * m.m33(),
                this.m10() * m.m00() + this.m11() * m.m10() + this.m12() * m.m20() + this.m13() * m.m30(),
                this.m10() * m.m01() + this.m11() * m.m11() + this.m12() * m.m21() + this.m13() * m.m31(),
                this.m10() * m.m02() + this.m11() * m.m12() + this.m12() * m.m22() + this.m13() * m.m32(),
                this.m10() * m.m03() + this.m11() * m.m13() + this.m12() * m.m23() + this.m13() * m.m33(),
                this.m20() * m.m00() + this.m21() * m.m10() + this.m22() * m.m20() + this.m23() * m.m30(),
                this.m20() * m.m01() + this.m21() * m.m11() + this.m22() * m.m21() + this.m23() * m.m31(),
                this.m20() * m.m02() + this.m21() * m.m12() + this.m22() * m.m22() + this.m23() * m.m32(),
                this.m20() * m.m03() + this.m21() * m.m13() + this.m22() * m.m23() + this.m23() * m.m33(),
                this.m30() * m.m00() + this.m31() * m.m10() + this.m32() * m.m20() + this.m33() * m.m30(),
                this.m30() * m.m01() + this.m31() * m.m11() + this.m32() * m.m21() + this.m33() * m.m31(),
                this.m30() * m.m02() + this.m31() * m.m12() + this.m32() * m.m22() + this.m33() * m.m32(),
                this.m30() * m.m03() + this.m31() * m.m13() + this.m32() * m.m23() + this.m33() * m.m33(),
                newType );
    },

    timesVector4: function( v ) {
      var x = this.m00() * v.x + this.m01() * v.y + this.m02() * v.z + this.m03() * v.w;
      var y = this.m10() * v.x + this.m11() * v.y + this.m12() * v.z + this.m13() * v.w;
      var z = this.m20() * v.x + this.m21() * v.y + this.m22() * v.z + this.m23() * v.w;
      var w = this.m30() * v.x + this.m31() * v.y + this.m32() * v.z + this.m33() * v.w;
      return new dot.Vector4( x, y, z, w );
    },

    timesVector3: function( v ) {
      return this.timesVector4( v.toVector4() ).toVector3();
    },

    timesTransposeVector4: function( v ) {
      var x = this.m00() * v.x + this.m10() * v.y + this.m20() * v.z + this.m30() * v.w;
      var y = this.m01() * v.x + this.m11() * v.y + this.m21() * v.z + this.m31() * v.w;
      var z = this.m02() * v.x + this.m12() * v.y + this.m22() * v.z + this.m32() * v.w;
      var w = this.m03() * v.x + this.m13() * v.y + this.m23() * v.z + this.m33() * v.w;
      return new dot.Vector4( x, y, z, w );
    },

    timesTransposeVector3: function( v ) {
      return this.timesTransposeVector4( v.toVector4() ).toVector3();
    },

    timesRelativeVector3: function( v ) {
      var x = this.m00() * v.x + this.m10() * v.y + this.m20() * v.z;
      var y = this.m01() * v.y + this.m11() * v.y + this.m21() * v.z;
      var z = this.m02() * v.z + this.m12() * v.y + this.m22() * v.z;
      return new dot.Vector3( x, y, z );
    },

    determinant: function() {
      return this.m03() * this.m12() * this.m21() * this.m30() -
          this.m02() * this.m13() * this.m21() * this.m30() -
          this.m03() * this.m11() * this.m22() * this.m30() +
          this.m01() * this.m13() * this.m22() * this.m30() +
          this.m02() * this.m11() * this.m23() * this.m30() -
          this.m01() * this.m12() * this.m23() * this.m30() -
          this.m03() * this.m12() * this.m20() * this.m31() +
          this.m02() * this.m13() * this.m20() * this.m31() +
          this.m03() * this.m10() * this.m22() * this.m31() -
          this.m00() * this.m13() * this.m22() * this.m31() -
          this.m02() * this.m10() * this.m23() * this.m31() +
          this.m00() * this.m12() * this.m23() * this.m31() +
          this.m03() * this.m11() * this.m20() * this.m32() -
          this.m01() * this.m13() * this.m20() * this.m32() -
          this.m03() * this.m10() * this.m21() * this.m32() +
          this.m00() * this.m13() * this.m21() * this.m32() +
          this.m01() * this.m10() * this.m23() * this.m32() -
          this.m00() * this.m11() * this.m23() * this.m32() -
          this.m02() * this.m11() * this.m20() * this.m33() +
          this.m01() * this.m12() * this.m20() * this.m33() +
          this.m02() * this.m10() * this.m21() * this.m33() -
          this.m00() * this.m12() * this.m21() * this.m33() -
          this.m01() * this.m10() * this.m22() * this.m33() +
          this.m00() * this.m11() * this.m22() * this.m33();
    },

    toString: function() {
      return this.m00() + " " + this.m01() + " " + this.m02() + " " + this.m03() + "\n" +
           this.m10() + " " + this.m11() + " " + this.m12() + " " + this.m13() + "\n" +
           this.m20() + " " + this.m21() + " " + this.m22() + " " + this.m23() + "\n" +
           this.m30() + " " + this.m31() + " " + this.m32() + " " + this.m33();
    },

    translation: function() { return new dot.Vector3( this.m03(), this.m13(), this.m23() ); },
    scaling: function() { return new dot.Vector3( this.m00(), this.m11(), this.m22() );},

    makeImmutable: function() {
      this.rowMajor = function() {
        throw new Error( "Cannot modify immutable matrix" );
      };
    }
  };

  // create an immutable
  Matrix4.IDENTITY = new Matrix4();
  Matrix4.IDENTITY.makeImmutable();
  
  return Matrix4;
} );

// Copyright 2002-2012, University of Colorado

/**
 * 3-dimensional Matrix
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Matrix3',['require','DOT/dot','DOT/Vector2','DOT/Vector3','DOT/Matrix4'], function( require ) {
  
  
  var dot = require( 'DOT/dot' );
  
  require( 'DOT/Vector2' );
  require( 'DOT/Vector3' );
  require( 'DOT/Matrix4' );
  
  dot.Matrix3 = function( v00, v01, v02, v10, v11, v12, v20, v21, v22, type ) {

    // entries stored in column-major format
    this.entries = new Array( 9 );

    this.rowMajor( v00 === undefined ? 1 : v00, v01 || 0, v02 || 0,
                   v10 || 0, v11 === undefined ? 1 : v11, v12 || 0,
                   v20 || 0, v21 || 0, v22 === undefined ? 1 : v22,
                   type );
  };
  var Matrix3 = dot.Matrix3;

  Matrix3.Types = {
    // NOTE: if an inverted matrix of a type is not that type, change inverted()!
    // NOTE: if two matrices with identical types are multiplied, the result should have the same type. if not, changed timesMatrix()!
    // NOTE: on adding a type, exaustively check all type usage
    OTHER: 0, // default
    IDENTITY: 1,
    TRANSLATION_2D: 2,
    SCALING: 3,
    AFFINE: 4

    // TODO: possibly add rotations
  };

  var Types = Matrix3.Types;

  Matrix3.identity = function() {
    return new Matrix3( 1, 0, 0,
                        0, 1, 0,
                        0, 0, 1,
                        Types.IDENTITY );
  };

  Matrix3.translation = function( x, y ) {
    return new Matrix3( 1, 0, x,
                        0, 1, y,
                        0, 0, 1,
                        Types.TRANSLATION_2D );
  };

  Matrix3.translationFromVector = function( v ) { return Matrix3.translation( v.x, v.y ); };

  Matrix3.scaling = function( x, y ) {
    // allow using one parameter to scale everything
    y = y === undefined ? x : y;

    return new Matrix3( x, 0, 0,
                        0, y, 0,
                        0, 0, 1,
                        Types.SCALING );
  };
  Matrix3.scale = Matrix3.scaling;

  // axis is a normalized Vector3, angle in radians.
  Matrix3.rotationAxisAngle = function( axis, angle ) {
    var c = Math.cos( angle );
    var s = Math.sin( angle );
    var C = 1 - c;

    return new Matrix3( axis.x * axis.x * C + c, axis.x * axis.y * C - axis.z * s, axis.x * axis.z * C + axis.y * s,
                        axis.y * axis.x * C + axis.z * s, axis.y * axis.y * C + c, axis.y * axis.z * C - axis.x * s,
                        axis.z * axis.x * C - axis.y * s, axis.z * axis.y * C + axis.x * s, axis.z * axis.z * C + c,
                        Types.OTHER );
  };

  // TODO: add in rotation from quaternion, and from quat + translation

  Matrix3.rotationX = function( angle ) {
    var c = Math.cos( angle );
    var s = Math.sin( angle );

    return new Matrix3( 1, 0, 0,
                        0, c, -s,
                        0, s, c,
                        Types.OTHER );
  };

  Matrix3.rotationY = function( angle ) {
    var c = Math.cos( angle );
    var s = Math.sin( angle );

    return new Matrix3( c, 0, s,
                        0, 1, 0,
                        -s, 0, c,
                        Types.OTHER );
  };

  Matrix3.rotationZ = function( angle ) {
    var c = Math.cos( angle );
    var s = Math.sin( angle );

    return new Matrix3( c, -s, 0,
                        s, c, 0,
                        0, 0, 1,
                        Types.AFFINE );
  };
  
  // standard 2d rotation
  Matrix3.rotation2 = Matrix3.rotationZ;
  
  Matrix3.fromSVGMatrix = function( svgMatrix ) {
    return new Matrix3( svgMatrix.a, svgMatrix.c, svgMatrix.e,
                        svgMatrix.b, svgMatrix.d, svgMatrix.f,
                        0, 0, 1,
                        Types.AFFINE );
  };

  // a rotation matrix that rotates A to B, by rotating about the axis A.cross( B ) -- Shortest path. ideally should be unit vectors
  Matrix3.rotateAToB = function( a, b ) {
    // see http://graphics.cs.brown.edu/~jfh/papers/Moller-EBA-1999/paper.pdf for information on this implementation
    var start = a;
    var end = b;

    var epsilon = 0.0001;

    var e, h, f;

    var v = start.cross( end );
    e = start.dot( end );
    f = ( e < 0 ) ? -e : e;

    // if "from" and "to" vectors are nearly parallel
    if ( f > 1.0 - epsilon ) {
      var c1, c2, c3;
      /* coefficients for later use */
      var i, j;

      var x = new dot.Vector3(
        ( start.x > 0.0 ) ? start.x : -start.x,
        ( start.y > 0.0 ) ? start.y : -start.y,
        ( start.z > 0.0 ) ? start.z : -start.z
      );

      if ( x.x < x.y ) {
        if ( x.x < x.z ) {
          x = dot.Vector3.X_UNIT;
        }
        else {
          x = dot.Vector3.Z_UNIT;
        }
      }
      else {
        if ( x.y < x.z ) {
          x = dot.Vector3.Y_UNIT;
        }
        else {
          x = dot.Vector3.Z_UNIT;
        }
      }

      var u = x.minus( start );
      v = x.minus( end );

      c1 = 2.0 / u.dot( u );
      c2 = 2.0 / v.dot( v );
      c3 = c1 * c2 * u.dot( v );

      return Matrix3.IDENTITY.plus( Matrix3.rowMajor(
        -c1 * u.x * u.x - c2 * v.x * v.x + c3 * v.x * u.x,
        -c1 * u.x * u.y - c2 * v.x * v.y + c3 * v.x * u.y,
        -c1 * u.x * u.z - c2 * v.x * v.z + c3 * v.x * u.z,
        -c1 * u.y * u.x - c2 * v.y * v.x + c3 * v.y * u.x,
        -c1 * u.y * u.y - c2 * v.y * v.y + c3 * v.y * u.y,
        -c1 * u.y * u.z - c2 * v.y * v.z + c3 * v.y * u.z,
        -c1 * u.z * u.x - c2 * v.z * v.x + c3 * v.z * u.x,
        -c1 * u.z * u.y - c2 * v.z * v.y + c3 * v.z * u.y,
        -c1 * u.z * u.z - c2 * v.z * v.z + c3 * v.z * u.z
      ) );
    }
    else {
      // the most common case, unless "start"="end", or "start"=-"end"
      var hvx, hvz, hvxy, hvxz, hvyz;
      h = 1.0 / ( 1.0 + e );
      hvx = h * v.x;
      hvz = h * v.z;
      hvxy = hvx * v.y;
      hvxz = hvx * v.z;
      hvyz = hvz * v.y;

      return Matrix3.rowMajor(
        e + hvx * v.x, hvxy - v.z, hvxz + v.y,
        hvxy + v.z, e + h * v.y * v.y, hvyz - v.x,
        hvxz - v.y, hvyz + v.x, e + hvz * v.z
      );
    }
  };

  Matrix3.prototype = {
    constructor: Matrix3,
    
    /*---------------------------------------------------------------------------*
    * "Properties"
    *----------------------------------------------------------------------------*/
    
    // convenience getters. inline usages of these when performance is critical? TODO: test performance of inlining these, with / without closure compiler
    m00: function() { return this.entries[0]; },
    m01: function() { return this.entries[3]; },
    m02: function() { return this.entries[6]; },
    m10: function() { return this.entries[1]; },
    m11: function() { return this.entries[4]; },
    m12: function() { return this.entries[7]; },
    m20: function() { return this.entries[2]; },
    m21: function() { return this.entries[5]; },
    m22: function() { return this.entries[8]; },
    
    isAffine: function() {
      return this.type === Types.AFFINE || ( this.m20() === 0 && this.m21() === 0 && this.m22() === 1 );
    },
    
    getDeterminant: function() {
      return this.m00() * this.m11() * this.m22() + this.m01() * this.m12() * this.m20() + this.m02() * this.m10() * this.m21() - this.m02() * this.m11() * this.m20() - this.m01() * this.m10() * this.m22() - this.m00() * this.m12() * this.m21();
    },
    get determinant() { return this.getDeterminant(); },
    
    getTranslation: function() {
      return new dot.Vector2( this.m02(), this.m12() );
    },
    get translation() { return this.getTranslation(); },
    
    // returns a vector that is equivalent to ( T(1,0).magnitude(), T(0,1).magnitude() ) where T is a relative transform
    getScaleVector: function() {
      return new dot.Vector2( Math.sqrt( this.m00() * this.m00() + this.m10() * this.m10() ),
                              Math.sqrt( this.m01() * this.m01() + this.m11() * this.m11() ) );
    },
    get scaleVector() { return this.getScaleVector(); },
    
    // angle in radians for the 2d rotation from this matrix, between pi, -pi
    getRotation: function() {
      var transformedVector = this.timesVector2( dot.Vector2.X_UNIT ).minus( this.timesVector2( dot.Vector2.ZERO ) );
      return Math.atan2( transformedVector.y, transformedVector.x );
    },
    get rotation() { return this.getRotation(); },
    
    toMatrix4: function() {
      return new dot.Matrix4( this.m00(), this.m01(), this.m02(), 0,
                              this.m10(), this.m11(), this.m12(), 0,
                              this.m20(), this.m21(), this.m22(), 0,
                              0, 0, 0, 1 );
    },
    
    toString: function() {
      return this.m00() + ' ' + this.m01() + ' ' + this.m02() + '\n' +
             this.m10() + ' ' + this.m11() + ' ' + this.m12() + '\n' +
             this.m20() + ' ' + this.m21() + ' ' + this.m22();
    },
    
    toSVGMatrix: function() {
      var result = document.createElementNS( 'http://www.w3.org/2000/svg', 'svg' ).createSVGMatrix();
      
      // top two rows
      result.a = this.m00();
      result.b = this.m10();
      result.c = this.m01();
      result.d = this.m11();
      result.e = this.m02();
      result.f = this.m12();
      
      return result;
    },
    
    getCSSTransform: function() {
      // See http://www.w3.org/TR/css3-transforms/, particularly Section 13 that discusses the SVG compatibility
      
      // we need to prevent the numbers from being in an exponential toString form, since the CSS transform does not support that
      function cssNumber( number ) {
        // largest guaranteed number of digits according to https://developer.mozilla.org/en-US/docs/JavaScript/Reference/Global_Objects/Number/toFixed
        return number.toFixed( 20 );
      }
      // the inner part of a CSS3 transform, but remember to add the browser-specific parts!
      return 'matrix(' + cssNumber( this.entries[0] ) + ',' + cssNumber( this.entries[1] ) + ',' + cssNumber( this.entries[3] ) + ',' + cssNumber( this.entries[4] ) + ',' + cssNumber( this.entries[6] ) + ',' + cssNumber( this.entries[7] ) + ')';
    },
    get cssTransform() { return this.getCSSTransform(); },
    
    getSVGTransform: function() {
      // SVG transform presentation attribute. See http://www.w3.org/TR/SVG/coords.html#TransformAttribute
      
      // we need to prevent the numbers from being in an exponential toString form, since the CSS transform does not support that
      function svgNumber( number ) {
        // largest guaranteed number of digits according to https://developer.mozilla.org/en-US/docs/JavaScript/Reference/Global_Objects/Number/toFixed
        return number.toFixed( 20 );
      }
      
      switch( this.type ) {
        case Types.IDENTITY:
          return '';
        case Types.TRANSLATION_2D:
          return 'translate(' + svgNumber( this.entries[6] ) + ',' + this.entries[7] + ')';
        case Types.SCALING:
          return 'scale(' + svgNumber( this.entries[0] ) + ( this.entries[0] === this.entries[4] ? '' : ',' + svgNumber( this.entries[4] ) ) + ')';
        default:
          return 'matrix(' + svgNumber( this.entries[0] ) + ',' + svgNumber( this.entries[1] ) + ',' + svgNumber( this.entries[3] ) + ',' + svgNumber( this.entries[4] ) + ',' + svgNumber( this.entries[6] ) + ',' + svgNumber( this.entries[7] ) + ')';
      }
    },
    get svgTransform() { return this.getSVGTransform(); },
    
    // returns a parameter object suitable for use with jQuery's .css()
    getCSSTransformStyles: function() {
      var transformCSS = this.getCSSTransform();
      
      // notes on triggering hardware acceleration: http://creativejs.com/2011/12/day-2-gpu-accelerate-your-dom-elements/
      return {
        // force iOS hardware acceleration
        '-webkit-perspective': 1000,
        '-webkit-backface-visibility': 'hidden',
        
        '-webkit-transform': transformCSS + ' translateZ(0)', // trigger hardware acceleration if possible
        '-moz-transform': transformCSS + ' translateZ(0)', // trigger hardware acceleration if possible
        '-ms-transform': transformCSS,
        '-o-transform': transformCSS,
        'transform': transformCSS,
        'transform-origin': 'top left', // at the origin of the component. consider 0px 0px instead. Critical, since otherwise this defaults to 50% 50%!!! see https://developer.mozilla.org/en-US/docs/CSS/transform-origin
        '-ms-transform-origin': 'top left' // TODO: do we need other platform-specific transform-origin styles?
      };
    },
    get cssTransformStyles() { return this.getCSSTransformStyles(); },
    
    // exact equality
    equals: function( m ) {
      return this.m00() === m.m00() && this.m01() === m.m01() && this.m02() === m.m02() &&
             this.m10() === m.m10() && this.m11() === m.m11() && this.m12() === m.m12() &&
             this.m20() === m.m20() && this.m21() === m.m21() && this.m22() === m.m22();
    },
    
    // equality within a margin of error
    equalsEpsilon: function( m, epsilon ) {
      return Math.abs( this.m00() - m.m00() ) < epsilon && Math.abs( this.m01() - m.m01() ) < epsilon && Math.abs( this.m02() - m.m02() ) < epsilon &&
             Math.abs( this.m10() - m.m10() ) < epsilon && Math.abs( this.m11() - m.m11() ) < epsilon && Math.abs( this.m12() - m.m12() ) < epsilon &&
             Math.abs( this.m20() - m.m20() ) < epsilon && Math.abs( this.m21() - m.m21() ) < epsilon && Math.abs( this.m22() - m.m22() ) < epsilon;
    },
    
    /*---------------------------------------------------------------------------*
    * Immutable operations (returns a new matrix)
    *----------------------------------------------------------------------------*/
    
    copy: function() {
      return new Matrix3(
        this.m00(), this.m01(), this.m02(),
        this.m10(), this.m11(), this.m12(),
        this.m20(), this.m21(), this.m22(),
        this.type
      );
    },
    
    plus: function( m ) {
      return new Matrix3(
        this.m00() + m.m00(), this.m01() + m.m01(), this.m02() + m.m02(),
        this.m10() + m.m10(), this.m11() + m.m11(), this.m12() + m.m12(),
        this.m20() + m.m20(), this.m21() + m.m21(), this.m22() + m.m22()
      );
    },
    
    minus: function( m ) {
      return new Matrix3(
        this.m00() - m.m00(), this.m01() - m.m01(), this.m02() - m.m02(),
        this.m10() - m.m10(), this.m11() - m.m11(), this.m12() - m.m12(),
        this.m20() - m.m20(), this.m21() - m.m21(), this.m22() - m.m22()
      );
    },
    
    transposed: function() {
      return new Matrix3(
        this.m00(), this.m10(), this.m20(),
        this.m01(), this.m11(), this.m21(),
        this.m02(), this.m12(), this.m22(), ( this.type === Types.IDENTITY || this.type === Types.SCALING ) ? this.type : undefined
      );
    },
    
    negated: function() {
      return new Matrix3(
        -this.m00(), -this.m01(), -this.m02(),
        -this.m10(), -this.m11(), -this.m12(),
        -this.m20(), -this.m21(), -this.m22()
      );
    },
    
    inverted: function() {
      var det;
      
      switch ( this.type ) {
        case Types.IDENTITY:
          return this;
        case Types.TRANSLATION_2D:
          return new Matrix3( 1, 0, -this.m02(),
                              0, 1, -this.m12(),
                              0, 0, 1, Types.TRANSLATION_2D );
        case Types.SCALING:
          return new Matrix3( 1 / this.m00(), 0, 0,
                              0, 1 / this.m11(), 0,
                              0, 0, 1 / this.m22(), Types.SCALING );
        case Types.AFFINE:
          det = this.getDeterminant();
          if ( det !== 0 ) {
            return new Matrix3(
              ( -this.m12() * this.m21() + this.m11() * this.m22() ) / det,
              ( this.m02() * this.m21() - this.m01() * this.m22() ) / det,
              ( -this.m02() * this.m11() + this.m01() * this.m12() ) / det,
              ( this.m12() * this.m20() - this.m10() * this.m22() ) / det,
              ( -this.m02() * this.m20() + this.m00() * this.m22() ) / det,
              ( this.m02() * this.m10() - this.m00() * this.m12() ) / det,
              0, 0, 1, Types.AFFINE
            );
          } else {
            throw new Error( 'Matrix could not be inverted, determinant === 0' );
          }
          break; // because JSHint totally can't tell that this can't be reached
        case Types.OTHER:
          det = this.getDeterminant();
          if ( det !== 0 ) {
            return new Matrix3(
              ( -this.m12() * this.m21() + this.m11() * this.m22() ) / det,
              ( this.m02() * this.m21() - this.m01() * this.m22() ) / det,
              ( -this.m02() * this.m11() + this.m01() * this.m12() ) / det,
              ( this.m12() * this.m20() - this.m10() * this.m22() ) / det,
              ( -this.m02() * this.m20() + this.m00() * this.m22() ) / det,
              ( this.m02() * this.m10() - this.m00() * this.m12() ) / det,
              ( -this.m11() * this.m20() + this.m10() * this.m21() ) / det,
              ( this.m01() * this.m20() - this.m00() * this.m21() ) / det,
              ( -this.m01() * this.m10() + this.m00() * this.m11() ) / det,
              Types.OTHER
            );
          } else {
            throw new Error( 'Matrix could not be inverted, determinant === 0' );
          }
          break; // because JSHint totally can't tell that this can't be reached
        default:
          throw new Error( 'Matrix3.inverted with unknown type: ' + this.type );
      }
    },
    
    timesMatrix: function( m ) {
      // I * M === M * I === I (the identity)
      if( this.type === Types.IDENTITY || m.type === Types.IDENTITY ) {
        return this.type === Types.IDENTITY ? m : this;
      }
      
      if ( this.type === m.type ) {
        // currently two matrices of the same type will result in the same result type
        if ( this.type === Types.TRANSLATION_2D ) {
          // faster combination of translations
          return new Matrix3( 1, 0, this.m02() + m.m02(),
                              0, 1, this.m12() + m.m12(),
                              0, 0, 1, Types.TRANSLATION_2D );
        } else if ( this.type === Types.SCALING ) {
          // faster combination of scaling
          return new Matrix3( this.m00() * m.m00(), 0, 0,
                              0, this.m11() * m.m11(), 0,
                              0, 0, 1, Types.SCALING );
        }
      }
      
      if ( this.type !== Types.OTHER && m.type !== Types.OTHER ) {
        // currently two matrices that are anything but "other" are technically affine, and the result will be affine
        
        // affine case
        return new Matrix3( this.m00() * m.m00() + this.m01() * m.m10(),
                            this.m00() * m.m01() + this.m01() * m.m11(),
                            this.m00() * m.m02() + this.m01() * m.m12() + this.m02(),
                            this.m10() * m.m00() + this.m11() * m.m10(),
                            this.m10() * m.m01() + this.m11() * m.m11(),
                            this.m10() * m.m02() + this.m11() * m.m12() + this.m12(),
                            0, 0, 1, Types.AFFINE );
      }
      
      // general case
      return new Matrix3( this.m00() * m.m00() + this.m01() * m.m10() + this.m02() * m.m20(),
                          this.m00() * m.m01() + this.m01() * m.m11() + this.m02() * m.m21(),
                          this.m00() * m.m02() + this.m01() * m.m12() + this.m02() * m.m22(),
                          this.m10() * m.m00() + this.m11() * m.m10() + this.m12() * m.m20(),
                          this.m10() * m.m01() + this.m11() * m.m11() + this.m12() * m.m21(),
                          this.m10() * m.m02() + this.m11() * m.m12() + this.m12() * m.m22(),
                          this.m20() * m.m00() + this.m21() * m.m10() + this.m22() * m.m20(),
                          this.m20() * m.m01() + this.m21() * m.m11() + this.m22() * m.m21(),
                          this.m20() * m.m02() + this.m21() * m.m12() + this.m22() * m.m22() );
    },
    
    /*---------------------------------------------------------------------------*
    * Immutable operations (returns new form of a parameter)
    *----------------------------------------------------------------------------*/
    
    timesVector2: function( v ) {
      var x = this.m00() * v.x + this.m01() * v.y + this.m02();
      var y = this.m10() * v.x + this.m11() * v.y + this.m12();
      return new dot.Vector2( x, y );
    },
    
    timesVector3: function( v ) {
      var x = this.m00() * v.x + this.m01() * v.y + this.m02() * v.z;
      var y = this.m10() * v.x + this.m11() * v.y + this.m12() * v.z;
      var z = this.m20() * v.x + this.m21() * v.y + this.m22() * v.z;
      return new dot.Vector3( x, y, z );
    },
    
    timesTransposeVector2: function( v ) {
      var x = this.m00() * v.x + this.m10() * v.y;
      var y = this.m01() * v.x + this.m11() * v.y;
      return new dot.Vector2( x, y );
    },
    
    // TODO: this operation seems to not work for transformDelta2, should be vetted
    timesRelativeVector2: function( v ) {
      var x = this.m00() * v.x + this.m01() * v.y;
      var y = this.m10() * v.y + this.m11() * v.y;
      return new dot.Vector2( x, y );
    },
    
    /*---------------------------------------------------------------------------*
    * Mutable operations (changes this matrix)
    *----------------------------------------------------------------------------*/
    
    makeImmutable: function() {
      this.rowMajor = function() {
        throw new Error( 'Cannot modify immutable matrix' );
      };
      return this;
    },
    
    rowMajor: function( v00, v01, v02, v10, v11, v12, v20, v21, v22, type ) {
      this.entries[0] = v00;
      this.entries[1] = v10;
      this.entries[2] = v20;
      this.entries[3] = v01;
      this.entries[4] = v11;
      this.entries[5] = v21;
      this.entries[6] = v02;
      this.entries[7] = v12;
      this.entries[8] = v22;
      
      // TODO: consider performance of the affine check here
      this.type = type === undefined ? ( ( v20 === 0 && v21 === 0 && v22 === 1 ) ? Types.AFFINE : Types.OTHER ) : type;
      return this;
    },
    
    columnMajor: function( v00, v10, v20, v01, v11, v21, v02, v12, v22, type ) {
      return this.rowMajor( v00, v01, v02, v10, v11, v12, v20, v21, v22, type );
    },
    
    add: function( m ) {
      return this.rowMajor(
        this.m00() + m.m00(), this.m01() + m.m01(), this.m02() + m.m02(),
        this.m10() + m.m10(), this.m11() + m.m11(), this.m12() + m.m12(),
        this.m20() + m.m20(), this.m21() + m.m21(), this.m22() + m.m22()
      );
    },
    
    subtract: function( m ) {
      return this.rowMajor(
        this.m00() - m.m00(), this.m01() - m.m01(), this.m02() - m.m02(),
        this.m10() - m.m10(), this.m11() - m.m11(), this.m12() - m.m12(),
        this.m20() - m.m20(), this.m21() - m.m21(), this.m22() - m.m22()
      );
    },
    
    transpose: function() {
      return this.rowMajor(
        this.m00(), this.m10(), this.m20(),
        this.m01(), this.m11(), this.m21(),
        this.m02(), this.m12(), this.m22(),
        ( this.type === Types.IDENTITY || this.type === Types.SCALING ) ? this.type : undefined
      );
    },
    
    negate: function() {
      return this.rowMajor(
        -this.m00(), -this.m01(), -this.m02(),
        -this.m10(), -this.m11(), -this.m12(),
        -this.m20(), -this.m21(), -this.m22()
      );
    },
    
    invert: function() {
      var det;
      
      switch ( this.type ) {
        case Types.IDENTITY:
          return this;
        case Types.TRANSLATION_2D:
          return this.rowMajor( 1, 0, -this.m02(),
                                0, 1, -this.m12(),
                                0, 0, 1, Types.TRANSLATION_2D );
        case Types.SCALING:
          return this.rowMajor( 1 / this.m00(), 0, 0,
                                0, 1 / this.m11(), 0,
                                0, 0, 1 / this.m22(), Types.SCALING );
        case Types.AFFINE:
          det = this.getDeterminant();
          if ( det !== 0 ) {
            return this.rowMajor(
              ( -this.m12() * this.m21() + this.m11() * this.m22() ) / det,
              ( this.m02() * this.m21() - this.m01() * this.m22() ) / det,
              ( -this.m02() * this.m11() + this.m01() * this.m12() ) / det,
              ( this.m12() * this.m20() - this.m10() * this.m22() ) / det,
              ( -this.m02() * this.m20() + this.m00() * this.m22() ) / det,
              ( this.m02() * this.m10() - this.m00() * this.m12() ) / det,
              0, 0, 1, Types.AFFINE
            );
          } else {
            throw new Error( 'Matrix could not be inverted, determinant === 0' );
          }
          break; // because JSHint totally can't tell that this can't be reached
        case Types.OTHER:
          det = this.getDeterminant();
          if ( det !== 0 ) {
            return this.rowMajor(
              ( -this.m12() * this.m21() + this.m11() * this.m22() ) / det,
              ( this.m02() * this.m21() - this.m01() * this.m22() ) / det,
              ( -this.m02() * this.m11() + this.m01() * this.m12() ) / det,
              ( this.m12() * this.m20() - this.m10() * this.m22() ) / det,
              ( -this.m02() * this.m20() + this.m00() * this.m22() ) / det,
              ( this.m02() * this.m10() - this.m00() * this.m12() ) / det,
              ( -this.m11() * this.m20() + this.m10() * this.m21() ) / det,
              ( this.m01() * this.m20() - this.m00() * this.m21() ) / det,
              ( -this.m01() * this.m10() + this.m00() * this.m11() ) / det,
              Types.OTHER
            );
          } else {
            throw new Error( 'Matrix could not be inverted, determinant === 0' );
          }
          break; // because JSHint totally can't tell that this can't be reached
        default:
          throw new Error( 'Matrix3.inverted with unknown type: ' + this.type );
      }
    },
    
    multiplyMatrix: function( m ) {
      // I * M === M * I === I (the identity)
      if( this.type === Types.IDENTITY || m.type === Types.IDENTITY ) {
        return this.type === Types.IDENTITY ? m : this;
      }
      
      if ( this.type === m.type ) {
        // currently two matrices of the same type will result in the same result type
        if ( this.type === Types.TRANSLATION_2D ) {
          // faster combination of translations
          return this.rowMajor( 1, 0, this.m02() + m.m02(),
                                0, 1, this.m12() + m.m12(),
                                0, 0, 1, Types.TRANSLATION_2D );
        } else if ( this.type === Types.SCALING ) {
          // faster combination of scaling
          return this.rowMajor( this.m00() * m.m00(), 0, 0,
                                0, this.m11() * m.m11(), 0,
                                0, 0, 1, Types.SCALING );
        }
      }
      
      if ( this.type !== Types.OTHER && m.type !== Types.OTHER ) {
        // currently two matrices that are anything but "other" are technically affine, and the result will be affine
        
        // affine case
        return this.rowMajor( this.m00() * m.m00() + this.m01() * m.m10(),
                              this.m00() * m.m01() + this.m01() * m.m11(),
                              this.m00() * m.m02() + this.m01() * m.m12() + this.m02(),
                              this.m10() * m.m00() + this.m11() * m.m10(),
                              this.m10() * m.m01() + this.m11() * m.m11(),
                              this.m10() * m.m02() + this.m11() * m.m12() + this.m12(),
                              0, 0, 1, Types.AFFINE );
      }
      
      // general case
      return this.rowMajor( this.m00() * m.m00() + this.m01() * m.m10() + this.m02() * m.m20(),
                            this.m00() * m.m01() + this.m01() * m.m11() + this.m02() * m.m21(),
                            this.m00() * m.m02() + this.m01() * m.m12() + this.m02() * m.m22(),
                            this.m10() * m.m00() + this.m11() * m.m10() + this.m12() * m.m20(),
                            this.m10() * m.m01() + this.m11() * m.m11() + this.m12() * m.m21(),
                            this.m10() * m.m02() + this.m11() * m.m12() + this.m12() * m.m22(),
                            this.m20() * m.m00() + this.m21() * m.m10() + this.m22() * m.m20(),
                            this.m20() * m.m01() + this.m21() * m.m11() + this.m22() * m.m21(),
                            this.m20() * m.m02() + this.m21() * m.m12() + this.m22() * m.m22() );
    },
    
    /*---------------------------------------------------------------------------*
    * Mutable operations (changes the parameter)
    *----------------------------------------------------------------------------*/
    
    multiplyVector2: function( v ) {
      var x = this.m00() * v.x + this.m01() * v.y + this.m02();
      var y = this.m10() * v.x + this.m11() * v.y + this.m12();
      v.setX( x );
      v.setY( y );
      return v;
    },
    
    multiplyVector3: function( v ) {
      var x = this.m00() * v.x + this.m01() * v.y + this.m02() * v.z;
      var y = this.m10() * v.x + this.m11() * v.y + this.m12() * v.z;
      var z = this.m20() * v.x + this.m21() * v.y + this.m22() * v.z;
      v.setX( x );
      v.setY( y );
      v.setZ( z );
      return v;
    },
    
    multiplyTransposeVector2: function( v ) {
      var x = this.m00() * v.x + this.m10() * v.y;
      var y = this.m01() * v.x + this.m11() * v.y;
      v.setX( x );
      v.setY( y );
      return v;
    },
    
    multiplyRelativeVector2: function( v ) {
      var x = this.m00() * v.x + this.m01() * v.y;
      var y = this.m10() * v.y + this.m11() * v.y;
      v.setX( x );
      v.setY( y );
      return v;
    },
    
    // sets the transform of a Canvas 2D rendering context to the affine part of this matrix
    canvasSetTransform: function( context ) {
      context.setTransform(
        // inlined array entries
        this.entries[0],
        this.entries[1],
        this.entries[3],
        this.entries[4],
        this.entries[6],
        this.entries[7]
      );
    },
    
    // appends the affine part of this matrix to the Canvas 2D rendering context
    canvasAppendTransform: function( context ) {
      if ( this.type !== Types.IDENTITY ) {
        context.transform(
          // inlined array entries
          this.entries[0],
          this.entries[1],
          this.entries[3],
          this.entries[4],
          this.entries[6],
          this.entries[7]
        );
      }
    }
  };

  // create an immutable
  Matrix3.IDENTITY = new Matrix3( 1, 0, 0,
                                  0, 1, 0,
                                  0, 0, 1,
                                  Types.IDENTITY );
  Matrix3.IDENTITY.makeImmutable();
  
  Matrix3.X_REFLECTION = new Matrix3( -1, 0, 0,
                                       0, 1, 0,
                                       0, 0, 1,
                                       Types.AFFINE );
  Matrix3.X_REFLECTION.makeImmutable();
  
  Matrix3.Y_REFLECTION = new Matrix3( 1,  0, 0,
                                      0, -1, 0,
                                      0,  0, 1,
                                      Types.AFFINE );
  Matrix3.Y_REFLECTION.makeImmutable();
  
  Matrix3.printer = {
    print: function( matrix ) {
      console.log( matrix.toString() );
    }
  };
  
  return Matrix3;
} );

// Copyright 2002-2012, University of Colorado

/**
 * 2-dimensional ray
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Ray2',['require','ASSERT/assert','DOT/dot'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'dot' );
  
  var dot = require( 'DOT/dot' );

  dot.Ray2 = function( pos, dir ) {
    this.pos = pos;
    this.dir = dir;
    
    assert && assert( Math.abs( dir.magnitude() - 1 ) < 0.01 );
  };
  var Ray2 = dot.Ray2;

  Ray2.prototype = {
    constructor: Ray2,

    shifted: function( distance ) {
      return new Ray2( this.pointAtDistance( distance ), this.dir );
    },

    pointAtDistance: function( distance ) {
      return this.pos.plus( this.dir.timesScalar( distance ) );
    },

    toString: function() {
      return this.pos.toString() + " => " + this.dir.toString();
    }
  };
  
  return Ray2;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Forward and inverse transforms with 3x3 matrices
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Transform3',['require','ASSERT/assert','DOT/dot','DOT/Matrix3','DOT/Vector2','DOT/Ray2'], function( require ) {
  

  var assert = require( 'ASSERT/assert' )( 'dot' );
  
  var dot = require( 'DOT/dot' );
  
  require( 'DOT/Matrix3' );
  require( 'DOT/Vector2' );
  require( 'DOT/Ray2' );

  // takes a 4x4 matrix
  dot.Transform3 = function( matrix ) {
    this.listeners = [];
    
    // using immutable version for now. change it to the mutable identity copy if we need mutable operations on the matrices
    this.set( matrix === undefined ? dot.Matrix3.IDENTITY : matrix );
  };
  var Transform3 = dot.Transform3;

  Transform3.prototype = {
    constructor: Transform3,
    
    /*---------------------------------------------------------------------------*
    * mutators
    *----------------------------------------------------------------------------*/
    
    set: function( matrix ) {
      assert && assert( matrix instanceof dot.Matrix3 );
      
      var oldMatrix = this.matrix;
      var length = this.listeners.length;
      var i;
      
      // notify listeners before the change
      for ( i = 0; i < length; i++ ) {
        this.listeners[i].before( matrix, oldMatrix );
      }
      
      this.matrix = matrix;
      
      // compute these lazily
      this.inverse = null;
      this.matrixTransposed = null;
      this.inverseTransposed = null;
      
      // notify listeners after the change
      for ( i = 0; i < length; i++ ) {
        this.listeners[i].after( matrix, oldMatrix );
      }
    },
    
    prepend: function( matrix ) {
      this.set( matrix.timesMatrix( this.matrix ) );
    },

    append: function( matrix ) {
      this.set( this.matrix.timesMatrix( matrix ) );
    },

    prependTransform: function( transform ) {
      this.prepend( transform.matrix );
    },

    appendTransform: function( transform ) {
      this.append( transform.matrix );
    },

    applyToCanvasContext: function( context ) {
      context.setTransform( this.matrix.m00(), this.matrix.m10(), this.matrix.m01(), this.matrix.m11(), this.matrix.m02(), this.matrix.m12() );
    },
    
    /*---------------------------------------------------------------------------*
    * getters
    *----------------------------------------------------------------------------*/
    
    // uses the same matrices, for use cases where the matrices are considered immutable
    copy: function() {
      var transform = new Transform3( this.matrix );
      transform.inverse = this.inverse;
      transform.matrixTransposed = this.matrixTransposed;
      transform.inverseTransposed = this.inverseTransposed;
    },
    
    // copies matrices, for use cases where the matrices are considered mutable
    deepCopy: function() {
      var transform = new Transform3( this.matrix.copy() );
      transform.inverse = this.inverse ? this.inverse.copy() : null;
      transform.matrixTransposed = this.matrixTransposed ? this.matrixTransposed.copy() : null;
      transform.inverseTransposed = this.inverseTransposed ? this.inverseTransposed.copy() : null;
    },
    
    getMatrix: function() {
      return this.matrix;
    },
    
    getInverse: function() {
      if ( this.inverse === null ) {
        this.inverse = this.matrix.inverted();
      }
      return this.inverse;
    },
    
    getMatrixTransposed: function() {
      if ( this.matrixTransposed === null ) {
        this.matrixTransposed = this.matrix.transposed();
      }
      return this.matrixTransposed;
    },
    
    getInverseTransposed: function() {
      if ( this.inverseTransposed === null ) {
        this.inverseTransposed = this.getInverse().transposed();
      }
      return this.inverseTransposed;
    },
    
    isIdentity: function() {
      return this.matrix.type === dot.Matrix3.Types.IDENTITY;
    },

    /*---------------------------------------------------------------------------*
     * forward transforms (for Vector2 or scalar)
     *----------------------------------------------------------------------------*/

    // transform a position (includes translation)
    transformPosition2: function( vec2 ) {
      return this.matrix.timesVector2( vec2 );
    },

    // transform a vector (exclude translation)
    transformDelta2: function( vec2 ) {
      // transform actually has the translation rolled into the other coefficients, so we have to make this longer
      return this.transformPosition2( vec2 ).minus( this.transformPosition2( dot.Vector2.ZERO ) );
    },

    // transform a normal vector (different than a normal vector)
    transformNormal2: function( vec2 ) {
      return this.getInverse().timesTransposeVector2( vec2 );
    },

    transformDeltaX: function( x ) {
      return this.transformDelta2( new dot.Vector2( x, 0 ) ).x;
    },

    transformDeltaY: function( y ) {
      return this.transformDelta2( new dot.Vector2( 0, y ) ).y;
    },
    
    transformBounds2: function( bounds2 ) {
      return bounds2.transformed( this.matrix );
    },
    
    transformShape: function( shape ) {
      return shape.transformed( this.matrix );
    },
    
    transformRay2: function( ray ) {
      return new dot.Ray2( this.transformPosition2( ray.pos ), this.transformDelta2( ray.dir ).normalized() );
    },

    /*---------------------------------------------------------------------------*
     * inverse transforms (for Vector2 or scalar)
     *----------------------------------------------------------------------------*/

    inversePosition2: function( vec2 ) {
      return this.getInverse().timesVector2( vec2 );
    },

    inverseDelta2: function( vec2 ) {
      // inverse actually has the translation rolled into the other coefficients, so we have to make this longer
      return this.inversePosition2( vec2 ).minus( this.inversePosition2( dot.Vector2.ZERO ) );
    },

    inverseNormal2: function( vec2 ) {
      return this.matrix.timesTransposeVector2( vec2 );
    },

    inverseDeltaX: function( x ) {
      return this.inverseDelta2( new dot.Vector2( x, 0 ) ).x;
    },

    inverseDeltaY: function( y ) {
      return this.inverseDelta2( new dot.Vector2( 0, y ) ).y;
    },
    
    inverseBounds2: function( bounds2 ) {
      return bounds2.transformed( this.getInverse() );
    },
    
    inverseShape: function( shape ) {
      return shape.transformed( this.getInverse() );
    },
    
    inverseRay2: function( ray ) {
      return new dot.Ray2( this.inversePosition2( ray.pos ), this.inverseDelta2( ray.dir ).normalized() );
    },
    
    /*---------------------------------------------------------------------------*
    * listeners
    *----------------------------------------------------------------------------*/
    
    // note: listener.before( matrix, oldMatrix ) will be called before the change, listener.after( matrix, oldMatrix ) will be called after
    addTransformListener: function( listener ) {
      assert && assert( !_.contains( this.listeners, listener ) );
      this.listeners.push( listener );
    },
    
    removeTransformListener: function( listener ) {
      assert && assert( _.contains( this.listeners, listener ) );
      this.listeners.splice( _.indexOf( this.listeners, listener ), 1 );
    }
  };
  
  return Transform3;
} );

// Copyright 2002-2012, University of Colorado

/**
 * A 2D rectangle-shaped bounded area (bounding box)
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Bounds2',['require','ASSERT/assert','DOT/dot','DOT/Vector2'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'dot' );
  
  var dot = require( 'DOT/dot' );
  
  require( 'DOT/Vector2' );
  
  // not using x,y,width,height so that it can handle infinity-based cases in a better way
  dot.Bounds2 = function( minX, minY, maxX, maxY ) {
    assert && assert( maxY !== undefined, 'Bounds2 requires 4 parameters' );
    this.minX = minX;
    this.minY = minY;
    this.maxX = maxX;
    this.maxY = maxY;
  };
  var Bounds2 = dot.Bounds2;

  Bounds2.prototype = {
    constructor: Bounds2,
    
    /*---------------------------------------------------------------------------*
    * Properties
    *----------------------------------------------------------------------------*/
    
    getWidth: function() { return this.maxX - this.minX; },
    get width() { return this.getWidth(); },
    
    getHeight: function() { return this.maxY - this.minY; },
    get height() { return this.getHeight(); },
    
    getX: function() { return this.minX; },
    get x() { return this.getX(); },
    
    getY: function() { return this.minY; },
    get y() { return this.getY(); },
    
    getCenterX: function() { return ( this.maxX + this.minX ) / 2; },
    get centerX() { return this.getCenterX(); },
    
    getCenterY: function() { return ( this.maxY + this.minY ) / 2; },
    get centerY() { return this.getCenterY(); },
    
    getMinX: function() { return this.minX; },
    getMinY: function() { return this.minY; },
    getMaxX: function() { return this.maxX; },
    getMaxY: function() { return this.maxY; },
    
    isEmpty: function() { return this.getWidth() < 0 || this.getHeight() < 0; },
    
    isFinite: function() {
      return isFinite( this.minX ) && isFinite( this.minY ) && isFinite( this.maxX ) && isFinite( this.maxY );
    },
    
    isValid: function() {
      return !this.isEmpty() && this.isFinite();
    },
    
    // whether the coordinates are inside the bounding box (or on the boundary)
    containsCoordinates: function( x, y ) {
      return this.minX <= x && x <= this.maxX && this.minY <= y && y <= this.maxY;
    },
    
    // whether the point is inside the bounding box (or on the boundary)
    containsPoint: function( point ) {
      return this.containsCoordinates( point.x, point.y );
    },
    
    // whether this bounding box completely contains the argument bounding box
    containsBounds: function( bounds ) {
      return this.minX <= bounds.minX && this.maxX >= bounds.maxX && this.minY <= bounds.minY && this.maxY >= bounds.maxY;
    },
    
    // whether the intersection is non-empty (if they share any part of a boundary, this will be true)
    intersectsBounds: function( bounds ) {
      // TODO: more efficient way of doing this?
      return !this.intersection( bounds ).isEmpty();
    },
    
    toString: function() {
      return '[x:(' + this.minX + ',' + this.maxX + '),y:(' + this.minY + ',' + this.maxY + ')]';
    },

    equals: function( other ) {
      return this.minX === other.minX && this.minY === other.minY && this.maxX === other.maxX && this.maxY === other.maxY;
    },
    
    /*---------------------------------------------------------------------------*
    * Immutable operations
    *----------------------------------------------------------------------------*/
    
    copy: function() {
      return new Bounds2( this.minX, this.minY, this.maxX, this.maxY );
    },
    
    // immutable operations (bounding-box style handling, so that the relevant bounds contain everything)
    union: function( bounds ) {
      return new Bounds2(
        Math.min( this.minX, bounds.minX ),
        Math.min( this.minY, bounds.minY ),
        Math.max( this.maxX, bounds.maxX ),
        Math.max( this.maxY, bounds.maxY )
      );
    },
    intersection: function( bounds ) {
      return new Bounds2(
        Math.max( this.minX, bounds.minX ),
        Math.max( this.minY, bounds.minY ),
        Math.min( this.maxX, bounds.maxX ),
        Math.min( this.maxY, bounds.maxY )
      );
    },
    // TODO: difference should be well-defined, but more logic is needed to compute
    
    withCoordinates: function( x, y ) {
      return new Bounds2(
        Math.min( this.minX, x ),
        Math.min( this.minY, y ),
        Math.max( this.maxX, x ),
        Math.max( this.maxY, y )
      );
    },
    
    // like a union with a point-sized bounding box
    withPoint: function( point ) {
      return this.withCoordinates( point.x, point.y );
    },
    
    withMinX: function( minX ) { return new Bounds2( minX, this.minY, this.maxX, this.maxY ); },
    withMinY: function( minY ) { return new Bounds2( this.minX, minY, this.maxX, this.maxY ); },
    withMaxX: function( maxX ) { return new Bounds2( this.minX, this.minY, maxX, this.maxY ); },
    withMaxY: function( maxY ) { return new Bounds2( this.minX, this.minY, this.maxX, maxY ); },
    
    // copy rounded to integral values, expanding where necessary
    roundedOut: function() {
      return new Bounds2(
        Math.floor( this.minX ),
        Math.floor( this.minY ),
        Math.ceil( this.maxX ),
        Math.ceil( this.maxY )
      );
    },
    
    // copy rounded to integral values, contracting where necessary
    roundedIn: function() {
      return new Bounds2(
        Math.ceil( this.minX ),
        Math.ceil( this.minY ),
        Math.floor( this.maxX ),
        Math.floor( this.maxY )
      );
    },
    
    // transform a bounding box.
    // NOTE that box.transformed( matrix ).transformed( inverse ) may be larger than the original box
    transformed: function( matrix ) {
      return this.copy().transform( matrix );
    },
    
    // returns copy expanded on all sides by length d
    dilated: function( d ) {
      return new Bounds2( this.minX - d, this.minY - d, this.maxX + d, this.maxY + d );
    },
    
    // returns copy contracted on all sides by length d
    eroded: function( d ) {
      return this.dilated( -d );
    },
    
    shiftedX: function( x ) {
      return new Bounds2( this.minX + x, this.minY, this.maxX + x, this.maxY );
    },
    
    shiftedY: function( y ) {
      return new Bounds2( this.minX, this.minY + y, this.maxX, this.maxY + y );
    },
    
    shifted: function( x, y ) {
      return new Bounds2( this.minX + x, this.minY + y, this.maxX + x, this.maxY + y );
    },
    
    /*---------------------------------------------------------------------------*
    * Mutable operations
    *----------------------------------------------------------------------------*/
    
    set: function( minX, minY, maxX, maxY ) {
      this.minX = minX;
      this.minY = minY;
      this.maxX = maxX;
      this.maxY = maxY;
      return this;
    },
    
    setBounds: function( bounds ) {
      return this.set( bounds.minX, bounds.minY, bounds.maxX, bounds.maxY );
    },
    
    // mutable union
    includeBounds: function( bounds ) {
      this.minX = Math.min( this.minX, bounds.minX );
      this.minY = Math.min( this.minY, bounds.minY );
      this.maxX = Math.max( this.maxX, bounds.maxX );
      this.maxY = Math.max( this.maxY, bounds.maxY );
      return this;
    },
    
    // mutable intersection
    constrainBounds: function( bounds ) {
      this.minX = Math.max( this.minX, bounds.minX );
      this.minY = Math.max( this.minY, bounds.minY );
      this.maxX = Math.min( this.maxX, bounds.maxX );
      this.maxY = Math.min( this.maxY, bounds.maxY );
      return this;
    },
    
    addCoordinates: function( x, y ) {
      this.minX = Math.min( this.minX, x );
      this.minY = Math.min( this.minY, y );
      this.maxX = Math.max( this.maxX, x );
      this.maxY = Math.max( this.maxY, y );
      return this;
    },
    
    addPoint: function( point ) {
      return this.addCoordinates( point.x, point.y );
    },
    
    setMinX: function( minX ) { this.minX = minX; return this; },
    setMinY: function( minY ) { this.minY = minY; return this; },
    setMaxX: function( maxX ) { this.maxX = maxX; return this; },
    setMaxY: function( maxY ) { this.maxY = maxY; return this; },
    
    // round to integral values, expanding where necessary
    roundOut: function() {
      this.minX = Math.floor( this.minX );
      this.minY = Math.floor( this.minY );
      this.maxX = Math.ceil( this.maxX );
      this.maxY = Math.ceil( this.maxY );
      return this;
    },
    
    // round to integral values, contracting where necessary
    roundIn: function() {
      this.minX = Math.ceil( this.minX );
      this.minY = Math.ceil( this.minY );
      this.maxX = Math.floor( this.maxX );
      this.maxY = Math.floor( this.maxY );
      return this;
    },
    
    // transform a bounding box.
    // NOTE that box.transformed( matrix ).transformed( inverse ) may be larger than the original box
    transform: function( matrix ) {
      // do nothing
      if ( this.isEmpty() ) {
        return this;
      }
      var minX = this.minX;
      var minY = this.minY;
      var maxX = this.maxX;
      var maxY = this.maxY;
      
      // using mutable vector so we don't create excessive instances of Vector2 during this
      // make sure all 4 corners are inside this transformed bounding box
      var vector = new dot.Vector2();
      this.setBounds( Bounds2.NOTHING );
      this.addPoint( matrix.multiplyVector2( vector.set( minX, minY ) ) );
      this.addPoint( matrix.multiplyVector2( vector.set( minX, maxY ) ) );
      this.addPoint( matrix.multiplyVector2( vector.set( maxX, minY ) ) );
      this.addPoint( matrix.multiplyVector2( vector.set( maxX, maxY ) ) );
      return this;
    },
    
    // expands on all sides by length d
    dilate: function( d ) {
      return this.set( this.minX - d, this.minY - d, this.maxX + d, this.maxY + d );
    },
    
    // contracts on all sides by length d
    erode: function( d ) {
      return this.dilate( -d );
    },
    
    shiftX: function( x ) {
      return this.setMinX( this.minX + x ).setMaxX( this.maxX + x );
    },
    
    shiftY: function( y ) {
      return this.setMinY( this.minY + y ).setMaxY( this.maxY + y );
    },
    
    shift: function( x, y ) {
      return this.shiftX( x ).shiftY( y );
    }
  };
  
  // specific bounds useful for operations
  Bounds2.EVERYTHING = new Bounds2( Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY );
  Bounds2.NOTHING = new Bounds2( Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY );
  
  return Bounds2;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Controls the underlying layer behavior around a node. The node's LayerStrategy's enter() and exit() will be
 * called in a depth-first order during the layer building process, and will modify a LayerBuilder to signal any
 * layer-specific signals.
 *
 * This generally ensures that a layer containing the proper renderer and settings to support its associated node
 * will be created.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/layers/LayerStrategy',['require','ASSERT/assert','SCENERY/scenery'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  /*
   * If the node specifies a renderer, we will always push a preferred type. That type will be fresh (if rendererOptions are specified), otherwise
   * the top matching preferred type for that renderer will be used. This allows us to always pop in the exit().
   *
   * Specified as such, since there is no needed shared state (we can have node.layerStrategy = scenery.LayerStrategy for many nodes)
   */
  scenery.LayerStrategy = {
    // true iff enter/exit will push/pop a layer type to the preferred stack. currently limited to only one layer type per level.
    hasPreferredLayerType: function( pointer, layerBuilder ) {
      return pointer.trail.lastNode().hasRenderer();
    },
    
    getPreferredLayerType: function( pointer, layerBuilder ) {
      assert && assert( this.hasPreferredLayerType( pointer, layerBuilder ) ); // sanity check
      
      var node = pointer.trail.lastNode();
      var preferredLayerType;
      
      if ( node.hasRendererLayerType() ) {
        preferredLayerType = node.getRendererLayerType();
      } else {
        preferredLayerType = layerBuilder.bestPreferredLayerTypeFor( [ node.getRenderer() ] );
        if ( !preferredLayerType ) {
          // there was no preferred layer type matching, just use the default
          preferredLayerType = node.getRenderer().defaultLayerType;
        }
      }
      
      return preferredLayerType;
    },
    
    enter: function( pointer, layerBuilder ) {
      var trail = pointer.trail;
      var node = trail.lastNode();
      var preferredLayerType;
      
      // if the node has a renderer, always push a layer type, so that we can pop on the exit() and ensure consistent behavior
      if ( node.hasRenderer() ) {
        preferredLayerType = this.getPreferredLayerType( pointer, layerBuilder );
        
        // push the preferred layer type
        layerBuilder.pushPreferredLayerType( preferredLayerType );
        if ( layerBuilder.getCurrentLayerType() !== preferredLayerType ) {
          layerBuilder.switchToType( pointer, preferredLayerType );
        }
      } else if ( node.isPainted() ) {
        // node doesn't specify a renderer, but isPainted.
        
        var supportedRenderers = node._supportedRenderers;
        var currentType = layerBuilder.getCurrentLayerType();
        preferredLayerType = layerBuilder.bestPreferredLayerTypeFor( supportedRenderers );
        
        // If any of the preferred types are compatible, use the top one. This allows us to support caching and hierarchical layer types
        if ( preferredLayerType ) {
          if ( currentType !== preferredLayerType ) {
            layerBuilder.switchToType( pointer, preferredLayerType );
          }
        } else {
          // if no preferred types are compatible, only switch if the current type is also incompatible
          if ( !currentType || !currentType.supportsNode( node ) ) {
            layerBuilder.switchToType( pointer, supportedRenderers[0].defaultLayerType );
          }
        }
      }
      
      if ( node.isLayerSplitBefore() || this.hasSplitFlags( node ) ) {
        layerBuilder.switchToType( pointer, layerBuilder.getCurrentLayerType() );
      }
      
      if ( node.isPainted() ) {
        // trigger actual layer creation if necessary (allow collapsing of layers otherwise)
        layerBuilder.markPainted( pointer );
      }
    },
    
    // afterSelf: function( trail, layerBuilder ) {
    //   // no-op, and possibly not used
    // },
    
    // betweenChildren: function( trail, layerBuilder ) {
    //   // no-op, and possibly not used
    // },
    
    exit: function( pointer, layerBuilder ) {
      var trail = pointer.trail;
      var node = trail.lastNode();
      
      if ( node.hasRenderer() ) {
        layerBuilder.popPreferredLayerType();
        
        // switch down to the next lowest preferred layer type, if any. if null, pass the null to switchToType
        // this allows us to not 'leak' the renderer information, and the temporary layer type is most likely collapsed and ignored
        // NOTE: disabled for now, since this prevents us from having adjacent children sharing the same layer type
        // if ( layerBuilder.getCurrentLayerType() !== layerBuilder.getPreferredLayerType() ) {
        //   layerBuilder.switchToType( pointer, layerBuilder.getPreferredLayerType() );
        // }
      }
      
      if ( node.isLayerSplitAfter() || this.hasSplitFlags( node ) ) {
        layerBuilder.switchToType( pointer, layerBuilder.getCurrentLayerType() );
      }
    },
    
    // whether splitting before and after the node is required
    hasSplitFlags: function( node ) {
      // currently, only enforce splitting if we are using CSS transforms
      var rendererOptions = node.getRendererOptions();
      return node.hasRenderer() && rendererOptions && (
        rendererOptions.cssTranslation ||
        rendererOptions.cssRotation ||
        rendererOptions.cssScale ||
        rendererOptions.cssTransform
      );
    }
  };
  var LayerStrategy = scenery.LayerStrategy;
  
  return LayerStrategy;
} );

// Copyright 2002-2012, University of Colorado

/**
 * A node for the Scenery scene graph. Supports general directed acyclic graphics (DAGs).
 * Handles multiple layers with assorted types (Canvas 2D, SVG, DOM, WebGL, etc.).
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/nodes/Node',['require','ASSERT/assert','DOT/Bounds2','DOT/Transform3','DOT/Matrix3','DOT/Util','SCENERY/scenery','SCENERY/layers/LayerStrategy'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Transform3 = require( 'DOT/Transform3' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var clamp = require( 'DOT/Util' ).clamp;
  
  var scenery = require( 'SCENERY/scenery' );
  var LayerStrategy = require( 'SCENERY/layers/LayerStrategy' ); // used to set the default layer strategy on the prototype
  // require( 'SCENERY/layers/Renderer' ); // commented out so Require.js doesn't balk at the circular dependency
  
  // TODO: FIXME: Why do I have to comment out this dependency?
  // require( 'SCENERY/util/Trail' );
  // require( 'SCENERY/util/TrailPointer' );
  
  var globalIdCounter = 1;
  
  /*
   * Available keys for use in the options parameter object for a vanilla Node (not inherited), in the order they are executed in:
   *
   * children:         A list of children to add (in order)
   * cursor:           Will display the specified CSS cursor when the mouse is over this Node or one of its descendents. The Scene needs to have input listeners attached with an initialize method first.
   * visible:          If false, this node (and its children) will not be displayed (or get input events)
   * pickable:         If false, this node (and its children) will not get input events
   * translation:      Sets the translation of the node to either the specified dot.Vector2 value, or the x,y values from an object (e.g. translation: { x: 1, y: 2 } )
   * x:                Sets the x-translation of the node
   * y:                Sets the y-translation of the node
   * rotation:         Sets the rotation of the node in radians
   * scale:            Sets the scale of the node. Supports either a number (same x-y scale), or a dot.Vector2 / object with ob.x and ob.y to set the scale for each axis independently
   * left:             Sets the x-translation so that the left (min X) of the bounding box (in the parent coordinate frame) is at the specified value
   * right:            Sets the x-translation so that the right (max X) of the bounding box (in the parent coordinate frame) is at the specified value
   * top:              Sets the y-translation so that the top (min Y) of the bounding box (in the parent coordinate frame) is at the specified value
   * bottom:           Sets the y-translation so that the bottom (min Y) of the bounding box (in the parent coordinate frame) is at the specified value
   * centerX:          Sets the x-translation so that the horizontal center of the bounding box (in the parent coordinate frame) is at the specified value
   * centerY:          Sets the y-translation so that the vertical center of the bounding box (in the parent coordinate frame) is at the specified value
   * renderer:         Forces Scenery to use the specific renderer (canvas/svg) to display this node (and if possible, children). Accepts both strings (e.g. 'canvas', 'svg', etc.) or actual Renderer objects (e.g. Renderer.Canvas, Renderer.SVG, etc.)
   * rendererOptions:  Parameter object that is passed to the created layer, and can affect how the layering process works.
   * layerSplit:       Forces a split between layers before and after this node (and its children) have been rendered. Useful for performance with Canvas-based renderers.
   * layerSplitBefore: Forces a split between layers before this node (and its children) have been rendered. Useful for performance with Canvas-based renderers.
   * layerSplitAfter:  Forces a split between layers after this node (and its children) have been rendered. Useful for performance with Canvas-based renderers.
   */
  scenery.Node = function Node( options ) {
    var self = this;
    
    // assign a unique ID to this node (allows trails to get a unique list of IDs)
    this._id = globalIdCounter++;
    
    // Whether this node (and its children) will be visible when the scene is updated. Visible nodes by default will not be pickable either
    this._visible = true;
    
    // Opacity from 0 to 1
    this._opacity = 1;
    
    // Whether hit testing will check for this node (and its children).
    this._pickable = true;
    
    // This node and all children will be clipped by this shape (in addition to any other clipping shapes).
    // The shape should be in the local coordinate frame
    this._clipShape = null;
    
    // the CSS cursor to be displayed over this node. null should be the default (inherit) value
    this._cursor = null;
    
    this._children = []; // ordered
    this._parents = []; // unordered
    
    /*
     * Set up the transform reference. we add a listener so that the transform itself can be modified directly
     * by reference, or node.transform = <transform> / node.setTransform() can be used to change the transform reference.
     * Both should trigger the necessary event notifications for Scenery to keep track internally.
     */
    this._transform = new Transform3();
    this._transformListener = {
      before: function() { self.beforeTransformChange(); },
      after: function() { self.afterTransformChange(); }
    };
    this._transform.addTransformListener( this._transformListener );
    
    this._inputListeners = []; // for user input handling (mouse/touch)
    this._eventListeners = []; // for internal events like paint invalidation, layer invalidation, etc.
    
    // TODO: add getter/setters that will be able to invalidate whether this node is under any pointers, etc.
    this._includeStrokeInHitRegion = false;
    
    // bounds handling
    this._bounds = Bounds2.NOTHING; // for this node and its children, in "parent" coordinates
    this._selfBounds = Bounds2.NOTHING; // just for this node, in "local" coordinates
    this._childBounds = Bounds2.NOTHING; // just for children, in "local" coordinates
    this._boundsDirty = true;
    this._selfBoundsDirty = this.isPainted();
    this._childBoundsDirty = true;
    
    // dirty region handling
    this._paintDirty = false;
    this._childPaintDirty = false;
    this._oldPaintMarked = false; // flag indicates the last rendered bounds of this node and all descendants are marked for a repaint already
    
    // what type of renderer should be forced for this node.
    this._renderer = null;
    this._rendererOptions = null; // options that will determine the layer type
    this._rendererLayerType = null; // cached layer type that is used by the LayerStrategy
    
    // whether layers should be split before and/or after this node. setting both will put this node and its children into a separate layer
    this._layerSplitBefore = false;
    this._layerSplitAfter = false;
    
    if ( options ) {
      this.mutate( options );
    }
  };
  var Node = scenery.Node;
  
  Node.prototype = {
    constructor: Node,
    
    insertChild: function( index, node ) {
      assert && assert( node !== null && node !== undefined, 'insertChild cannot insert a null/undefined child' );
      assert && assert( !_.contains( this._children, node ), 'Parent already contains child' );
      assert && assert( node !== this, 'Cannot add self as a child' );
      
      node._parents.push( this );
      this._children.splice( index, 0, node );
      
      node.invalidateBounds();
      node.invalidatePaint();
      
      this.dispatchEvent( 'markForInsertion', {
        parent: this,
        child: node,
        index: index
      } );
      
      this.dispatchEvent( 'stitch', { match: false } );
    },
    
    addChild: function( node ) {
      this.insertChild( this._children.length, node );
    },
    
    removeChild: function( node ) {
      assert && assert( this.isChild( node ) );
      
      node.markOldPaint();
      
      var indexOfParent = _.indexOf( node._parents, this );
      var indexOfChild = _.indexOf( this._children, node );
      
      this.dispatchEvent( 'markForRemoval', {
        parent: this,
        child: node,
        index: indexOfChild
      } );
      
      node._parents.splice( indexOfParent, 1 );
      this._children.splice( indexOfChild, 1 );
      
      this.invalidateBounds();
      
      this.dispatchEvent( 'stitch', { match: false } );
    },
    
    // TODO: efficiency by batching calls?
    setChildren: function( children ) {
      var node = this;
      if ( this._children !== children ) {
        _.each( this._children.slice( 0 ), function( child ) {
          node.removeChild( child );
        } );
        _.each( children, function( child ) {
          node.addChild( child );
        } );
      }
    },
    
    getChildren: function() {
      return this._children.slice( 0 ); // create a defensive copy
    },
    
    getParents: function() {
      return this._parents.slice( 0 ); // create a defensive copy
    },
    
    indexOfParent: function( parent ) {
      return _.indexOf( this._parents, parent );
    },
    
    indexOfChild: function( child ) {
      return _.indexOf( this._children, child );
    },
    
    moveToFront: function() {
      var self = this;
      _.each( this._parents.slice( 0 ), function( parent ) {
        parent.moveChildToFront( self );
      } );
    },
    
    moveChildToFront: function( child ) {
      if ( this.indexOfChild( child ) !== this._children.length - 1 ) {
        this.removeChild( child );
        this.addChild( child );
      }
    },
    
    moveToBack: function() {
      var self = this;
      _.each( this._parents.slice( 0 ), function( parent ) {
        parent.moveChildToBack( self );
      } );
    },
    
    moveChildToBack: function( child ) {
      if ( this.indexOfChild( child ) !== 0 ) {
        this.removeChild( child );
        this.insertChild( 0, child );
      }
    },
    
    // remove this node from its parents
    detach: function() {
      var that = this;
      _.each( this._parents.slice( 0 ), function( parent ) {
        parent.removeChild( that );
      } );
    },
    
    // ensure that cached bounds stored on this node (and all children) are accurate
    validateBounds: function() {
      var that = this;
      
      if ( this._selfBoundsDirty ) {
        // note: this should only be triggered if the bounds were actually changed, since we have a guard in place at invalidateSelf()
        this._selfBoundsDirty = false;
        
        // if our self bounds changed, make sure to paint the area where our new bounds are
        this.markDirtyRegion( this._selfBounds );
        
        // TODO: consider changing to parameter object (that may be a problem for the GC overhead)
        this.fireEvent( 'selfBounds', this._selfBounds );
      }
      
      // validate bounds of children if necessary
      if ( this._childBoundsDirty ) {
        // have each child validate their own bounds
        _.each( this._children, function( child ) {
          child.validateBounds();
        } );
        
        var oldChildBounds = this._childBounds;
        
        // and recompute our _childBounds
        this._childBounds = Bounds2.NOTHING;
        
        _.each( this._children, function( child ) {
          that._childBounds = that._childBounds.union( child._bounds );
        } );
        
        this._childBoundsDirty = false;
        
        if ( !this._childBounds.equals( oldChildBounds ) ) {
          // TODO: consider changing to parameter object (that may be a problem for the GC overhead)
          this.fireEvent( 'childBounds', this._childBounds );
        }
      }
      
      // TODO: layout here?
      
      if ( this._boundsDirty ) {
        var oldBounds = this._bounds;
        
        var newBounds = this.localToParentBounds( this._selfBounds ).union( that.localToParentBounds( this._childBounds ) );
        var changed = !newBounds.equals( oldBounds );
        
        if ( changed ) {
          this._bounds = newBounds;
          
          _.each( this._parents, function( parent ) {
            parent.invalidateBounds();
          } );
          
          // TODO: consider changing to parameter object (that may be a problem for the GC overhead)
          this.fireEvent( 'bounds', this._bounds );
        }
        
        this._boundsDirty = false;
      }
    },
    
    validatePaint: function() {
      // if dirty, mark the region
      if ( this._paintDirty ) {
        this.markDirtyRegion( this.parentToLocalBounds( this._bounds ) );
        this._paintDirty = false;
      }
      
      // clear flags and recurse
      if ( this._childPaintDirty || this._oldPaintMarked ) {
        this._childPaintDirty = false;
        this._oldPaintMarked = false;
        
        _.each( this._children, function( child ) {
          child.validatePaint();
        } );
      }
    },
    
    // mark the bounds of this node as invalid, so it is recomputed before it is accessed again
    invalidateBounds: function() {
      this._boundsDirty = true;
      
      // and set flags for all ancestors
      _.each( this._parents, function( parent ) {
        parent.invalidateChildBounds();
      } );
    },
    
    // recursively tag all ancestors with _childBoundsDirty
    invalidateChildBounds: function() {
      // don't bother updating if we've already been tagged
      if ( !this._childBoundsDirty ) {
        this._childBoundsDirty = true;
        _.each( this._parents, function( parent ) {
          parent.invalidateChildBounds();
        } );
      }
    },
    
    // mark the paint of this node as invalid, so its new region will be painted
    invalidatePaint: function() {
      this._paintDirty = true;
      
      // and set flags for all ancestors
      _.each( this._parents, function( parent ) {
        parent.invalidateChildPaint();
      } );
    },
    
    // recursively tag all ancestors with _childPaintDirty
    invalidateChildPaint: function() {
      // don't bother updating if we've already been tagged
      if ( !this._childPaintDirty ) {
        this._childPaintDirty = true;
        _.each( this._parents, function( parent ) {
          parent.invalidateChildPaint();
        } );
      }
    },
    
    // called to notify that self rendering will display different paint, with possibly different bounds
    invalidateSelf: function( newBounds ) {
      assert && assert( newBounds.isEmpty() || newBounds.isFinite() , "Bounds must be empty or finite in invalidateSelf");
      
      // mark the old region to be repainted, regardless of whether the actual bounds change
      this.markOldSelfPaint();
      
      // if these bounds are different than current self bounds
      if ( !this._selfBounds.equals( newBounds ) ) {
        // set repaint flags
        this._selfBoundsDirty = true;
        this.invalidateBounds();
        
        // record the new bounds
        this._selfBounds = newBounds;
      }
      
      this.invalidatePaint();
    },
    
    // bounds assumed to be in the local coordinate frame, below this node's transform
    markDirtyRegion: function( bounds ) {
      this.dispatchEventWithTransform( 'dirtyBounds', {
        node: this,
        bounds: bounds
      } );
    },
    
    markOldSelfPaint: function() {
      this.markOldPaint( true );
    },
    
    // should be called whenever something triggers changes for how this node is layered
    markLayerRefreshNeeded: function() {
      this.dispatchEvent( 'markForLayerRefresh', {} );
      
      this.dispatchEvent( 'stitch', { match: true } );
    },
    
    // marks the last-rendered bounds of this node and optionally all of its descendants as needing a repaint
    markOldPaint: function( justSelf ) {
      function ancestorHasOldPaint( node ) {
        if( node._oldPaintMarked ) {
          return true;
        }
        return _.some( node._parents, function( parent ) {
          return ancestorHasOldPaint( parent );
        } );
      }
      
      var alreadyMarked = ancestorHasOldPaint( this );
      
      // we want to not do this marking if possible multiple times for the same sub-tree, so we check flags first
      if ( !alreadyMarked ) {
        if ( justSelf ) {
          this.markDirtyRegion( this._selfBounds );
        } else {
          this.markDirtyRegion( this.parentToLocalBounds( this._bounds ) );
          this._oldPaintMarked = true; // don't mark this in self calls, because we don't use the full bounds
        }
      }
    },
    
    isChild: function( potentialChild ) {
      var ourChild = _.contains( this._children, potentialChild );
      var itsParent = _.contains( potentialChild._parents, this );
      assert && assert( ourChild === itsParent );
      return ourChild;
    },
    
    // the bounds for self content in "local" coordinates
    getSelfBounds: function() {
      return this._selfBounds;
    },
    
    getChildBounds: function() {
      this.validateBounds();
      return this._childBounds;
    },
    
    // the bounds for content in render(), in "parent" coordinates
    getBounds: function() {
      this.validateBounds();
      return this._bounds;
    },
    
    /*
     * Return a trail to the top node (if any, otherwise null) whose self-rendered area contains the
     * point (in parent coordinates).
     *
     * If options.pruneInvisible is false, invisible nodes will be allowed in the trail.
     * If options.pruneUnpickable is false, unpickable nodes will be allowed in the trail.
     */
    trailUnderPoint: function( point, options ) {
      assert && assert( point, 'trailUnderPointer requires a point' );
      
      var pruneInvisible = ( !options || options.pruneInvisible === undefined ) ? true : options.pruneInvisible;
      var pruneUnpickable = ( !options || options.pruneUnpickable === undefined ) ? true : options.pruneUnpickable;
      
      if ( pruneInvisible && !this.isVisible() ) {
        return null;
      }
      if ( pruneUnpickable && !this.isPickable() ) {
        return null;
      }
      
      // update bounds for pruning
      this.validateBounds();
      
      // bail quickly if this doesn't hit our computed bounds
      if ( !this._bounds.containsPoint( point ) ) { return null; }
      
      // point in the local coordinate frame. computed after the main bounds check, so we can bail out there efficiently
      var localPoint = this._transform.inversePosition2( point );
      
      // check children first, since they are rendered later
      if ( this._children.length > 0 && this._childBounds.containsPoint( localPoint ) ) {
        
        // manual iteration here so we can return directly, and so we can iterate backwards (last node is in front)
        for ( var i = this._children.length - 1; i >= 0; i-- ) {
          var child = this._children[i];
          
          var childHit = child.trailUnderPoint( localPoint, options );
          
          // the child will have the point in its parent's coordinate frame (i.e. this node's frame)
          if ( childHit ) {
            childHit.addAncestor( this, i );
            return childHit;
          }
        }
      }
      
      // didn't hit our children, so check ourself as a last resort
      if ( this._selfBounds.containsPoint( localPoint ) && this.containsPointSelf( localPoint ) ) {
        return new scenery.Trail( this );
      }
      
      // signal no hit
      return null;
    },
    
    // checking for whether a point (in parent coordinates) is contained in this sub-tree
    containsPoint: function( point ) {
      return this.trailUnderPoint( point ) !== null;
    },
    
    // override for computation of whether a point is inside the self content
    // point is considered to be in the local coordinate frame
    containsPointSelf: function( point ) {
      // if self bounds are not null default to checking self bounds
      return this._selfBounds.containsPoint( point );
    },
    
    // whether this node's self intersects the specified bounds, in the local coordinate frame
    intersectsBoundsSelf: function( bounds ) {
      // if self bounds are not null, child should override this
      return this._selfBounds.intersectsBounds( bounds );
    },
    
    isPainted: function() {
      return false;
    },
    
    hasParent: function() {
      return this._parents.length !== 0;
    },
    
    hasChildren: function() {
      return this._children.length > 0;
    },
    
    walkDepthFirst: function( callback ) {
      callback( this );
      _.each( this._children, function( child ) {
        child.walkDepthFirst( callback );
      } );
    },
    
    getChildrenWithinBounds: function( bounds ) {
      return _.filter( this._children, function( child ) { return !child._bounds.intersection( bounds ).isEmpty(); } );
    },
    
    // TODO: set this up with a mix-in for a generic notifier?
    addInputListener: function( listener ) {
      // don't allow listeners to be added multiple times
      if ( _.indexOf( this._inputListeners, listener ) === -1 ) {
        this._inputListeners.push( listener );
      }
    },
    
    removeInputListener: function( listener ) {
      // ensure the listener is in our list
      assert && assert( _.indexOf( this._inputListeners, listener ) !== -1 );
      
      this._inputListeners.splice( _.indexOf( this._inputListeners, listener ), 1 );
    },
    
    getInputListeners: function() {
      return this._inputListeners.slice( 0 ); // defensive copy
    },
    
    // TODO: set this up with a mix-in for a generic notifier?
    addEventListener: function( listener ) {
      // don't allow listeners to be added multiple times
      if ( _.indexOf( this._eventListeners, listener ) === -1 ) {
        this._eventListeners.push( listener );
      }
    },
    
    removeEventListener: function( listener ) {
      // ensure the listener is in our list
      assert && assert( _.indexOf( this._eventListeners, listener ) !== -1 );
      
      this._eventListeners.splice( _.indexOf( this._eventListeners, listener ), 1 );
    },
    
    getEventListeners: function() {
      return this._eventListeners.slice( 0 ); // defensive copy
    },
    
    /*
     * Fires an event to all event listeners attached to this node. It does not bubble down to
     * all ancestors with trails, like dispatchEvent does. Use fireEvent when you only want an event
     * that is relevant for a specific node, and ancestors don't need to be notified.
     */
    fireEvent: function( type, args ) {
      _.each( this.getEventListeners(), function( eventListener ) {
        if ( eventListener[type] ) {
          eventListener[type]( args );
        }
      } );
    },
    
    /*
     * Dispatches an event across all possible Trails ending in this node.
     *
     * For example, if the scene has two children A and B, and both of those nodes have X as a child,
     * dispatching an event on X will fire the event with the following trails:
     * on X     with trail [ X ]
     * on A     with trail [ A, X ]
     * on scene with trail [ scene, A, X ]
     * on B     with trail [ B, X ]
     * on scene with trail [ scene, B, X ]
     *
     * This allows you to add a listener on any node to get notifications for all of the trails that the
     * event is relevant for (e.g. marks dirty paint region for both places X was on the scene).
     */
    dispatchEvent: function( type, args ) {
      var trail = new scenery.Trail();
      
      function recursiveEventDispatch( node ) {
        trail.addAncestor( node );
        
        args.trail = trail;
        
        node.fireEvent( type, args );
        
        _.each( node._parents, function( parent ) {
          recursiveEventDispatch( parent );
        } );
        
        trail.removeAncestor();
      }
      
      recursiveEventDispatch( this );
    },
    
    // dispatches events with the transform computed from parent of the "root" to the local frame
    dispatchEventWithTransform: function( type, args ) {
      var trail = new scenery.Trail();
      var transformStack = [ new Transform3() ];
      
      function recursiveEventDispatch( node ) {
        trail.addAncestor( node );
        
        transformStack.push( new Transform3( node.getMatrix().timesMatrix( transformStack[transformStack.length-1].getMatrix() ) ) );
        args.transform = transformStack[transformStack.length-1];
        args.trail = trail;
        
        node.fireEvent( type, args );
        
        _.each( node._parents, function( parent ) {
          recursiveEventDispatch( parent );
        } );
        
        transformStack.pop();
        
        trail.removeAncestor();
      }
      
      recursiveEventDispatch( this );
    },
    
    // TODO: consider renaming to translateBy to match scaleBy
    translate: function( x, y, prependInstead ) {
      if ( typeof x === 'number' ) {
        // translate( x, y, prependInstead )
        if ( prependInstead ) {
          this.prependMatrix( Matrix3.translation( x, y ) );
        } else {
          this.appendMatrix( Matrix3.translation( x, y ) );
        }
      } else {
        // translate( vector, prependInstead )
        var vector = x;
        this.translate( vector.x, vector.y, y ); // forward to full version
      }
    },
    
    // scale( s ) is also supported, which will scale both dimensions by the same amount. renamed from 'scale' to satisfy the setter/getter
    scale: function( x, y, prependInstead ) {
      if ( typeof x === 'number' ) {
        if ( y === undefined ) {
          // scale( scale )
          this.appendMatrix( Matrix3.scaling( x, x ) );
        } else {
          // scale( x, y, prependInstead )
          if ( prependInstead ) {
            this.prependMatrix( Matrix3.scaling( x, y ) );
          } else {
            this.appendMatrix( Matrix3.scaling( x, y ) );
          }
        }
      } else {
        // scale( vector, prependInstead ) or scale( { x: x, y: y }, prependInstead )
        var vector = x;
        this.scale( vector.x, vector.y, y ); // forward to full version
      }
    },
    
    // TODO: consider naming to rotateBy to match scaleBy (due to scale property / method name conflict)
    rotate: function( angle, prependInstead ) {
      if ( prependInstead ) {
        this.prependMatrix( Matrix3.rotation2( angle ) );
      } else {
        this.appendMatrix( Matrix3.rotation2( angle ) );
      }
    },
    
    // point should be in the parent coordinate frame
    // TODO: determine whether this should use the appendMatrix method
    rotateAround: function( point, angle ) {
      var matrix = Matrix3.translation( -point.x, -point.y );
      matrix = Matrix3.rotation2( angle ).timesMatrix( matrix );
      matrix = Matrix3.translation( point.x, point.y ).timesMatrix( matrix );
      this.prependMatrix( matrix );
    },
    
    getX: function() {
      return this.getTranslation().x;
    },
    
    setX: function( x ) {
      this.setTranslation( x, this.getY() );
      return this;
    },
    
    getY: function() {
      return this.getTranslation().y;
    },
    
    setY: function( y ) {
      this.setTranslation( this.getX(), y );
      return this;
    },
    
    // returns a vector with an entry for each axis, e.g. (5,2) for an Affine-style matrix with rows ((5,0,0),(0,2,0),(0,0,1))
    getScaleVector: function() {
      return this._transform.getMatrix().getScaleVector();
    },
    
    // supports setScaleMagnitude( 5 ) for both dimensions, setScaleMagnitude( 5, 3 ) for each dimension separately, or setScaleMagnitude( new Vector2( x, y ) )
    setScaleMagnitude: function( a, b ) {
      var currentScale = this.getScaleVector();
      
      if ( typeof a === 'number' ) {
        if ( b === undefined ) {
          // to map setScaleMagnitude( scale ) => setScaleMagnitude( scale, scale )
          b = a;
        }
        // setScaleMagnitude( x, y )
        this.appendMatrix( Matrix3.scaling( a / currentScale.x, b / currentScale.y ) );
      } else {
        // setScaleMagnitude( vector ), where we set the x-scale to vector.x and y-scale to vector.y
        this.appendMatrix( Matrix3.scaling( a.x / currentScale.x, a.y / currentScale.y ) );
      }
      return this;
    },
    
    getRotation: function() {
      return this._transform.getMatrix().getRotation();
    },
    
    setRotation: function( rotation ) {
      this.appendMatrix( Matrix3.rotation2( rotation - this.getRotation() ) );
      return this;
    },
    
    // supports setTranslation( x, y ) or setTranslation( new Vector2( x, y ) ) .. or technically setTranslation( { x: x, y: y } )
    setTranslation: function( a, b ) {
      var translation = this.getTranslation();
      
      if ( typeof a === 'number' ) {
        this.translate( a - translation.x, b - translation.y, true );
      } else {
        this.translate( a.x - translation.x, a.y - translation.y, true );
      }
      return this;
    },
    
    getTranslation: function() {
      return this._transform.getMatrix().getTranslation();
    },
    
    // append a transformation matrix to our local transform
    appendMatrix: function( matrix ) {
      this._transform.append( matrix );
    },
    
    // prepend a transformation matrix to our local transform
    prependMatrix: function( matrix ) {
      this._transform.prepend( matrix );
    },
    
    setMatrix: function( matrix ) {
      this._transform.set( matrix );
    },
    
    getMatrix: function() {
      return this._transform.getMatrix();
    },
    
    // change the actual transform reference (not just the actual transform)
    setTransform: function( transform ) {
      if ( this._transform !== transform ) {
        // since our referenced transform doesn't change, we need to trigger the before/after ourselves
        this.beforeTransformChange();
        
        // swap the transform and move the listener to the new one
        this._transform.removeTransformListener( this._transformListener ); // don't leak memory!
        this._transform = transform;
        this._transform.addTransformListener( this._transformListener );
        
        this.afterTransformChange();
      }
    },
    
    getTransform: function() {
      // for now, return an actual copy. we can consider listening to changes in the future
      return this._transform;
    },
    
    resetTransform: function() {
      this.setMatrix( Matrix3.IDENTITY );
    },
    
    // called before our transform is changed
    beforeTransformChange: function() {
      // mark our old bounds as dirty, so that any dirty region repainting will include not just our new position, but also our old position
      this.markOldPaint();
    },
    
    // called after our transform is changed
    afterTransformChange: function() {
      this.dispatchEventWithTransform( 'transform', {
        node: this,
        type: 'transform',
        matrix: this._transform.getMatrix()
      } );
      this.invalidateBounds();
      this.invalidatePaint();
    },
    
    // the left bound of this node, in the parent coordinate frame
    getLeft: function() {
      return this.getBounds().minX;
    },
    
    // shifts this node horizontally so that its left bound (in the parent coordinate frame) is 'left'
    setLeft: function( left ) {
      this.translate( left - this.getLeft(), 0, true );
      return this; // allow chaining
    },
    
    // the right bound of this node, in the parent coordinate frame
    getRight: function() {
      return this.getBounds().maxX;
    },
    
    // shifts this node horizontally so that its right bound (in the parent coordinate frame) is 'right'
    setRight: function( right ) {
      this.translate( right - this.getRight(), 0, true );
      return this; // allow chaining
    },
    
    getCenterX: function() {
      return this.getBounds().getCenterX();
    },
    
    setCenterX: function( x ) {
      this.translate( x - this.getCenterX(), 0, true );
      return this; // allow chaining
    },
    
    getCenterY: function() {
      return this.getBounds().getCenterY();
    },
    
    setCenterY: function( y ) {
      this.translate( 0, y - this.getCenterY(), true );
      return this; // allow chaining
    },
    
    // the top bound of this node, in the parent coordinate frame
    getTop: function() {
      return this.getBounds().minY;
    },
    
    // shifts this node vertically so that its top bound (in the parent coordinate frame) is 'top'
    setTop: function( top ) {
      this.translate( 0, top - this.getTop(), true );
      return this; // allow chaining
    },
    
    // the bottom bound of this node, in the parent coordinate frame
    getBottom: function() {
      return this.getBounds().maxY;
    },
    
    // shifts this node vertically so that its bottom bound (in the parent coordinate frame) is 'bottom'
    setBottom: function( bottom ) {
      this.translate( 0, bottom - this.getBottom(), true );
      return this; // allow chaining
    },
    
    getWidth: function() {
      return this.getBounds().getWidth();
    },
    
    getHeight: function() {
      return this.getBounds().getHeight();
    },
    
    getId: function() {
      return this._id;
    },
    
    isVisible: function() {
      return this._visible;
    },
    
    setVisible: function( visible ) {
      if ( visible !== this._visible ) {
        if ( this._visible ) {
          this.markOldSelfPaint();
        }
        
        this._visible = visible;
        
        this.invalidatePaint();
      }
      return this;
    },
    
    getOpacity: function() {
      return this._opacity;
    },
    
    setOpacity: function( opacity ) {
      var clampedOpacity = clamp( opacity, 0, 1 );
      if ( clampedOpacity !== this._opacity ) {
        this.markOldPaint();
        
        this._opacity = clampedOpacity;
        
        this.invalidatePaint();
      }
    },
    
    isPickable: function() {
      return this._pickable;
    },
    
    setPickable: function( pickable ) {
      if ( this._pickable !== pickable ) {
        // no paint or invalidation changes for now, since this is only handled for the mouse
        this._pickable = pickable;
        
        // TODO: invalidate the cursor somehow?
      }
    },
    
    setCursor: function( cursor ) {
      // TODO: consider a mapping of types to set reasonable defaults
      /*
      auto default none inherit help pointer progress wait crosshair text vertical-text alias copy move no-drop not-allowed
      e-resize n-resize w-resize s-resize nw-resize ne-resize se-resize sw-resize ew-resize ns-resize nesw-resize nwse-resize
      context-menu cell col-resize row-resize all-scroll url( ... ) --> does it support data URLs?
       */
      
      // allow the 'auto' cursor type to let the ancestors or scene pick the cursor type
      this._cursor = cursor === "auto" ? null : cursor;
    },
    
    getCursor: function() {
      return this._cursor;
    },
    
    updateLayerType: function() {
      if ( this._renderer && this._rendererOptions ) {
        // TODO: factor this check out! Make RendererOptions its own class?
        // TODO: FIXME: support undoing this!
        // ensure that if we are passing a CSS transform, we pass this node as the baseNode
        if ( this._rendererOptions.cssTransform || this._rendererOptions.cssTranslation || this._rendererOptions.cssRotation || this._rendererOptions.cssScale ) {
          this._rendererOptions.baseNode = this;
        } else if ( this._rendererOptions.hasOwnProperty( 'baseNode' ) ) {
          delete this._rendererOptions.baseNode; // don't override, let the scene pass in the scene
        }
        // if we set renderer and rendererOptions, only then do we want to trigger a specific layer type
        this._rendererLayerType = this._renderer.createLayerType( this._rendererOptions );
      } else {
        this._rendererLayerType = null; // nothing signaled, since we want to support multiple layer types (including if we specify a renderer)
      }
    },
    
    getRendererLayerType: function() {
      return this._rendererLayerType;
    },
    
    hasRendererLayerType: function() {
      return !!this._rendererLayerType;
    },
    
    setRenderer: function( renderer ) {
      var newRenderer;
      if ( typeof renderer === 'string' ) {
        assert && assert( scenery.Renderer[renderer], 'unknown renderer in setRenderer: ' + renderer );
        newRenderer = scenery.Renderer[renderer];
      } else if ( renderer instanceof scenery.Renderer ) {
        newRenderer = renderer;
      } else if ( !renderer ) {
        newRenderer = null;
      } else {
        throw new Error( 'unrecognized type of renderer: ' + renderer );
      }
      if ( newRenderer !== this._renderer ) {
        assert && assert( !this.isPainted() || !newRenderer || _.contains( this._supportedRenderers, newRenderer ), 'renderer ' + newRenderer + ' not supported by ' + this.constructor.name );
        this._renderer = newRenderer;
        
        this.updateLayerType();
        this.markLayerRefreshNeeded();
      }
    },
    
    getRenderer: function() {
      return this._renderer;
    },
    
    hasRenderer: function() {
      return !!this._renderer;
    },
    
    setRendererOptions: function( options ) {
      // TODO: consider checking options based on the specified 'renderer'?
      this._rendererOptions = options;
      
      this.updateLayerType();
      this.markLayerRefreshNeeded();
    },
    
    getRendererOptions: function() {
      return this._rendererOptions;
    },
    
    hasRendererOptions: function() {
      return !!this._rendererOptions;
    },
    
    setLayerSplitBefore: function( split ) {
      if ( this._layerSplitBefore !== split ) {
        this._layerSplitBefore = split;
        this.markLayerRefreshNeeded();
      }
    },
    
    isLayerSplitBefore: function() {
      return this._layerSplitBefore;
    },
    
    setLayerSplitAfter: function( split ) {
      if ( this._layerSplitAfter !== split ) {
        this._layerSplitAfter = split;
        this.markLayerRefreshNeeded();
      }
    },
    
    isLayerSplitAfter: function() {
      return this._layerSplitAfter;
    },
    
    setLayerSplit: function( split ) {
      if ( split !== this._layerSplitBefore || split !== this._layerSplitAfter ) {
        this._layerSplitBefore = split;
        this._layerSplitAfter = split;
        this.markLayerRefreshNeeded();
      }
    },
    
    // returns a unique trail (if it exists) where each node in the ancestor chain has 0 or 1 parents
    getUniqueTrail: function() {
      var trail = new scenery.Trail();
      var node = this;
      
      while ( node ) {
        trail.addAncestor( node );
        assert && assert( node._parents.length <= 1 );
        node = node._parents[0]; // should be undefined if there aren't any parents
      }
      
      return trail;
    },
    
    debugText: function() {
      var startPointer = new scenery.TrailPointer( new scenery.Trail( this ), true );
      var endPointer = new scenery.TrailPointer( new scenery.Trail( this ), false );
      
      var depth = 0;
      
      startPointer.depthFirstUntil( endPointer, function( pointer ) {
        if ( pointer.isBefore ) {
          // hackish way of multiplying a string
          var padding = new Array( depth * 2 ).join( ' ' );
          console.log( padding + pointer.trail.lastNode().getId() + ' ' + pointer.trail.toString() );
        }
        depth += pointer.isBefore ? 1 : -1;
      }, false );
    },
    
    /*
     * Renders this node to a canvas. If toCanvas( callback ) is used, the canvas will contain the node's
     * entire bounds.
     *
     * callback( canvas, x, y ) is called, where x and y offsets are computed if not specified.
     */
    toCanvas: function( callback, x, y, width, height ) {
      var self = this;
      
      var padding = 2; // padding used if x and y are not set
      
      var bounds = this.getBounds();
      x = x !== undefined ? x : Math.ceil( padding - bounds.minX );
      y = y !== undefined ? y : Math.ceil( padding - bounds.minY );
      width = width !== undefined ? width : Math.ceil( x + bounds.getWidth() + padding );
      height = height !== undefined ? height : Math.ceil( y + bounds.getHeight() + padding );
      
      var canvas = document.createElement( 'canvas' );
      canvas.width = width;
      canvas.height = height;
      var context = canvas.getContext( '2d' );
      
      var $div = $( document.createElement( 'div' ) );
      $div.width( width ).height( height );
      var scene = new scenery.Scene( $div );
      
      scene.addChild( self );
      scene.x = x;
      scene.y = y;
      scene.updateScene();
      
      scene.renderToCanvas( canvas, context, function() {
        callback( canvas, x, y );
        
        // let us be garbage collected
        scene.removeChild( self );
      } );
    },
    
    // gives a data URI, with the same parameter handling as Node.toCanvas()
    toDataURL: function( callback, x, y, width, height ) {
      this.toCanvas( function( canvas, x, y ) {
        // this x and y shadow the outside parameters, and will be different if the outside parameters are undefined
        callback( canvas.toDataURL(), x, y );
      }, x, y, width, height );
    },
    
    // gives an HTMLImageElement with the same parameter handling as Node.toCanvas()
    toImage: function( callback, x, y, width, height ) {
      this.toDataURL( function( url, x, y ) {
        // this x and y shadow the outside parameters, and will be different if the outside parameters are undefined
        var img = document.createElement( 'img' );
        img.onload = function() {
          callback( img, x, y );
          delete img.onload;
        };
        img.src = url;
      }, x, y, width, height );
    },
    
    /*---------------------------------------------------------------------------*
    * Coordinate transform methods
    *----------------------------------------------------------------------------*/
    
    // apply this node's transform to the point
    localToParentPoint: function( point ) {
      return this._transform.transformPosition2( point );
    },
    
    localToParentBounds: function( bounds ) {
      return this._transform.transformBounds2( bounds );
    },
    
    // apply the inverse of this node's transform to the point
    parentToLocalPoint: function( point ) {
      return this._transform.inversePosition2( point );
    },
    
    parentToLocalBounds: function( bounds ) {
      return this._transform.inverseBounds2( bounds );
    },
    
    // apply this node's transform (and then all of its parents' transforms) to the point
    localToGlobalPoint: function( point ) {
      var node = this;
      while ( node !== null ) {
        point = node._transform.transformPosition2( point );
        assert && assert( node._parents[1] === undefined, 'localToGlobalPoint unable to work for DAG' );
        node = node._parents[0];
      }
      return point;
    },
    
    localToGlobalBounds: function( bounds ) {
      var node = this;
      while ( node !== null ) {
        bounds = node._transform.transformBounds2( bounds );
        assert && assert( node._parents[1] === undefined, 'localToGlobalBounds unable to work for DAG' );
        node = node._parents[0];
      }
      return bounds;
    },
    
    globalToLocalPoint: function( point ) {
      var node = this;
      
      // we need to apply the transformations in the reverse order, so we temporarily store them
      var transforms = [];
      while ( node !== null ) {
        transforms.push( node._transform );
        assert && assert( node._parents[1] === undefined, 'globalToLocalPoint unable to work for DAG' );
        node = node._parents[0];
      }
      
      // iterate from the back forwards (from the root node to here)
      for ( var i = transforms.length - 1; i >=0; i-- ) {
        point = transforms[i].inversePosition2( point );
      }
      return point;
    },
    
    globalToLocalBounds: function( bounds ) {
      var node = this;
      
      // we need to apply the transformations in the reverse order, so we temporarily store them
      var transforms = [];
      while ( node !== null ) {
        transforms.push( node._transform );
        assert && assert( node._parents[1] === undefined, 'globalToLocalBounds unable to work for DAG' );
        node = node._parents[0];
      }
      
      // iterate from the back forwards (from the root node to here)
      for ( var i = transforms.length - 1; i >=0; i-- ) {
        bounds = transforms[i].inverseBounds2( bounds );
      }
      return bounds;
    },
    
    /*---------------------------------------------------------------------------*
    * ES5 get/set
    *----------------------------------------------------------------------------*/
    
    set layerSplit( value ) { this.setLayerSplit( value ); },
    get layerSplit() { throw new Error( 'You can\'t get a layerSplit property, since it modifies two separate properties' ); },
    
    set layerSplitBefore( value ) { this.setLayerSplitBefore( value ); },
    get layerSplitBefore() { return this.isLayerSplitBefore(); },
    
    set layerSplitAfter( value ) { this.setLayerSplitAfter( value ); },
    get layerSplitAfter() { return this.isLayerSplitAfter(); },
    
    set renderer( value ) { this.setRenderer( value ); },
    get renderer() { return this.getRenderer(); },
    
    set rendererOptions( value ) { this.setRendererOptions( value ); },
    get rendererOptions() { return this.getRendererOptions(); },
    
    set cursor( value ) { this.setCursor( value ); },
    get cursor() { return this.getCursor(); },
    
    set visible( value ) { this.setVisible( value ); },
    get visible() { return this.isVisible(); },
    
    set opacity( value ) { this.setOpacity( value ); },
    get opacity() { return this.getOpacity(); },
    
    set pickable( value ) { this.setPickable( value ); },
    get pickable() { return this.isPickable(); },
    
    set transform( value ) { this.setTransform( value ); },
    get transform() { return this.getTransform(); },
    
    set matrix( value ) { this.setMatrix( value ); },
    get matrix() { return this.getMatrix(); },
    
    set translation( value ) { this.setTranslation( value ); },
    get translation() { return this.getTranslation(); },
    
    set rotation( value ) { this.setRotation( value ); },
    get rotation() { return this.getRotation(); },
    
    set x( value ) { this.setX( value ); },
    get x() { return this.getX(); },
    
    set y( value ) { this.setY( value ); },
    get y() { return this.getY(); },
    
    set left( value ) { this.setLeft( value ); },
    get left() { return this.getLeft(); },
    
    set right( value ) { this.setRight( value ); },
    get right() { return this.getRight(); },
    
    set top( value ) { this.setTop( value ); },
    get top() { return this.getTop(); },
    
    set bottom( value ) { this.setBottom( value ); },
    get bottom() { return this.getBottom(); },
    
    set centerX( value ) { this.setCenterX( value ); },
    get centerX() { return this.getCenterX(); },
    
    set centerY( value ) { this.setCenterY( value ); },
    get centerY() { return this.getCenterY(); },
    
    set children( value ) { this.setChildren( value ); },
    get children() { return this.getChildren(); },
    
    get parents() { return this.getParents(); },
    
    get width() { return this.getWidth(); },
    get height() { return this.getHeight(); },
    get bounds() { return this.getBounds(); },
    get selfBounds() { return this.getSelfBounds(); },
    get childBounds() { return this.getChildBounds(); },
    get id() { return this.getId(); },
    
    mutate: function( options ) {
      var node = this;
      
      _.each( this._mutatorKeys, function( key ) {
        if ( options[key] !== undefined ) {
          var descriptor = Object.getOwnPropertyDescriptor( Node.prototype, key );
          
          // if the key refers to a function that is not ES5 writable, it will execute that function with the single argument
          if ( descriptor && typeof descriptor.value === 'function' ) {
            node[key]( options[key] );
          } else {
            node[key] = options[key];
          }
        }
      } );
      
      return this; // allow chaining
    },
    
    toString: function( spaces ) {
      spaces = spaces || '';
      var props = this.getPropString( spaces + '  ' );
      return spaces + this.getBasicConstructor( props ? ( '\n' + props + '\n' + spaces ) : '' );
    },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Node( {' + propLines + '} )';
    },
    
    getPropString: function( spaces ) {
      var self = this;
      
      var result = '';
      function addProp( key, value, nowrap ) {
        if ( result ) {
          result += ',\n';
        }
        if ( !nowrap && typeof value === 'string' ) {
          result += spaces + key + ': \'' + value + '\'';
        } else {
          result += spaces + key + ': ' + value;
        }
      }
      
      if ( this._children.length ) {
        var childString = '';
        _.each( this._children, function( child ) {
          if ( childString ) {
            childString += ',\n';
          }
          childString += child.toString( spaces + '  ' );
        } );
        addProp( 'children', '[\n' + childString + '\n' + spaces + ']', true );
      }
      
      // direct copy props
      if ( this.cursor ) { addProp( 'cursor', this.cursor ); }
      if ( !this.visible ) { addProp( 'visible', this.visible ); }
      if ( !this.pickable ) { addProp( 'pickable', this.pickable ); }
      if ( this.opacity !== 1 ) { addProp( 'opacity', this.opacity ); }
      
      if ( !this.transform.isIdentity() ) {
        var m = this.transform.getMatrix();
        addProp( 'matrix', 'new dot.Matrix3( ' + m.m00() + ', ' + m.m01() + ', ' + m.m02() + ', ' +
                                                 m.m10() + ', ' + m.m11() + ', ' + m.m12() + ', ' +
                                                 m.m20() + ', ' + m.m21() + ', ' + m.m22() + ' )', true );
      }
      
      if ( this.renderer ) {
        addProp( 'renderer', this.renderer.name );
        if ( this.rendererOptions ) {
          addProp( 'rendererOptions', JSON.stringify( this.rendererOptions ), true );
        }
      }
      
      if ( this._layerSplitBefore ) {
        addProp( 'layerSplitBefore', true );
      }
      
      if ( this._layerSplitAfter ) {
        addProp( 'layerSplitAfter', true );
      }
      
      return result;
    }
  };
  
  /*
   * This is an array of property (setter) names for Node.mutate(), which are also used when creating nodes with parameter objects.
   *
   * E.g. new scenery.Node( { x: 5, rotation: 20 } ) will create a Path, and apply setters in the order below (node.x = 5; node.rotation = 20)
   *
   * The order below is important! Don't change this without knowing the implications.
   * NOTE: translation-based mutators come before rotation/scale, since typically we think of their operations occuring "after" the rotation / scaling
   * NOTE: left/right/top/bottom/centerX/centerY are at the end, since they rely potentially on rotation / scaling changes of bounds that may happen beforehand
   * TODO: using more than one of {translation,x,left,right,centerX} or {translation,y,top,bottom,centerY} should be considered an error
   * TODO: move fill / stroke setting to mixins
   */
  Node.prototype._mutatorKeys = [ 'children', 'cursor', 'visible', 'pickable', 'opacity', 'matrix', 'translation', 'x', 'y', 'rotation', 'scale',
                                  'left', 'right', 'top', 'bottom', 'centerX', 'centerY', 'renderer', 'rendererOptions',
                                  'layerSplit', 'layerSplitBefore', 'layerSplitAfter' ];
  
  Node.prototype._supportedRenderers = [];
  
  Node.prototype.layerStrategy = LayerStrategy;
  
  return Node;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Represents a trail (path in the graph) from a "root" node down to a descendant node.
 * In a DAG, or with different views, there can be more than one trail up from a node,
 * even to the same root node!
 *
 * This trail also mimics an Array, so trail[0] will be the root, and trail[trail.length-1]
 * will be the end node of the trail.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/util/Trail',['require','ASSERT/assert','ASSERT/assert','DOT/Transform3','SCENERY/scenery','SCENERY/nodes/Node'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  var assertExtra = require( 'ASSERT/assert' )( 'scenery.extra', false );
  
  var Transform3 = require( 'DOT/Transform3' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  require( 'SCENERY/nodes/Node' );
  // require( 'SCENERY/util/TrailPointer' );
  
  scenery.Trail = function( nodes ) {
    if ( nodes instanceof Trail ) {
      // copy constructor (takes advantage of already built index information)
      var otherTrail = nodes;
      
      this.nodes = otherTrail.nodes.slice( 0 );
      this.length = otherTrail.length;
      this.indices = otherTrail.indices.slice( 0 );
      return;
    }
    
    this.nodes = [];
    this.length = 0;
    
    // indices[x] stores the index of nodes[x] in nodes[x-1]'s children
    this.indices = [];
    
    var trail = this;
    if ( nodes ) {
      if ( nodes instanceof scenery.Node ) {
        var node = nodes;
        
        // add just a single node in
        trail.addDescendant( node );
      } else {
        // process it as an array
        _.each( nodes, function( node ) {
          trail.addDescendant( node );
        } );
      }
    }
  };
  var Trail = scenery.Trail;
  
  Trail.prototype = {
    constructor: Trail,
    
    copy: function() {
      return new Trail( this );
    },
    
    // convenience function to determine whether this trail will render something
    isPainted: function() {
      return this.lastNode().isPainted();
    },
    
    get: function( index ) {
      if ( index >= 0 ) {
        return this.nodes[index];
      } else {
        // negative index goes from the end of the array
        return this.nodes[this.nodes.length + index];
      }
    },
    
    slice: function( startIndex, endIndex ) {
      return new Trail( this.nodes.slice( startIndex, endIndex ) );
    },
    
    subtrailTo: function( node, excludeNode ) {
      return this.slice( 0, _.indexOf( this.nodes, node ) + ( excludeNode ? 0 : 1 ) );
    },
    
    isEmpty: function() {
      return this.nodes.length === 0;
    },
    
    getTransform: function() {
      // always return a defensive copy of a transform
      var transform = new Transform3();
      
      // from the root up
      _.each( this.nodes, function( node ) {
        transform.appendTransform( node._transform );
      } );
      
      return transform;
    },
    
    addAncestor: function( node, index ) {
      var oldRoot = this.nodes[0];
      
      this.nodes.unshift( node );
      if ( oldRoot ) {
        this.indices.unshift( index === undefined ? _.indexOf( node._children, oldRoot ) : index );
      }
      
      // mimic an Array
      this.length++;
      return this;
    },
    
    removeAncestor: function() {
      this.nodes.shift();
      if ( this.indices.length ) {
        this.indices.shift();
      }
      
      // mimic an Array
      this.length--;
      return this;
    },
    
    addDescendant: function( node, index ) {
      var parent = this.lastNode();
      
      this.nodes.push( node );
      if ( parent ) {
        this.indices.push( index === undefined ? _.indexOf( parent._children, node ) : index );
      }
      
      // mimic an Array
      this.length++;
      return this;
    },
    
    removeDescendant: function() {
      this.nodes.pop();
      if ( this.indices.length ) {
        this.indices.pop();
      }
      
      // mimic an Array
      this.length--;
      return this;
    },
    
    // refreshes the internal index references (important if any children arrays were modified!)
    reindex: function() {
      for ( var i = 1; i < this.length; i++ ) {
        // only replace indices where they have changed (this was a performance hotspot)
        var currentIndex = this.indices[i-1];
        if ( this.nodes[i-1]._children[currentIndex] !== this.nodes[i] ) {
          this.indices[i-1] = _.indexOf( this.nodes[i-1]._children, this.nodes[i] );
        }
      }
    },
    
    areIndicesValid: function() {
      for ( var i = 1; i < this.length; i++ ) {
        var currentIndex = this.indices[i-1];
        if ( this.nodes[i-1]._children[currentIndex] !== this.nodes[i] ) {
          return false;
        }
      }
      return true;
    },
    
    equals: function( other ) {
      if ( this.length !== other.length ) {
        return false;
      }
      
      for ( var i = 0; i < this.nodes.length; i++ ) {
        if ( this.nodes[i] !== other.nodes[i] ) {
          return false;
        }
      }
      
      return true;
    },
    
    // whether this trail contains the complete 'other' trail, but with added descendants afterwards
    isExtensionOf: function( other, allowSameTrail ) {
      assertExtra && assertExtra( this.areIndicesValid(), 'Trail.compare this.areIndicesValid() failed' );
      assertExtra && assertExtra( other.areIndicesValid(), 'Trail.compare other.areIndicesValid() failed' );
      
      if ( this.length <= other.length - ( allowSameTrail ? 1 : 0 ) ) {
        return false;
      }
      
      for ( var i = 0; i < other.nodes.length; i++ ) {
        if ( this.nodes[i] !== other.nodes[i] ) {
          return false;
        }
      }
      
      return true;
    },
    
    // TODO: phase out in favor of get()
    nodeFromTop: function( offset ) {
      return this.nodes[this.length - 1 - offset];
    },
    
    lastNode: function() {
      return this.nodeFromTop( 0 );
    },
    
    rootNode: function() {
      return this.nodes[0];
    },
    
    // returns the previous graph trail in the order of self-rendering
    previous: function() {
      if ( this.nodes.length <= 1 ) {
        return null;
      }
      
      var top = this.nodeFromTop( 0 );
      var parent = this.nodeFromTop( 1 );
      
      var parentIndex = _.indexOf( parent._children, top );
      assert && assert( parentIndex !== -1 );
      var arr = this.nodes.slice( 0, this.nodes.length - 1 );
      if ( parentIndex === 0 ) {
        // we were the first child, so give it the trail to the parent
        return new Trail( arr );
      } else {
        // previous child
        arr.push( parent._children[parentIndex-1] );
        
        // and find its last terminal
        while( arr[arr.length-1]._children.length !== 0 ) {
          var last = arr[arr.length-1];
          arr.push( last._children[last._children.length-1] );
        }
        
        return new Trail( arr );
      }
    },
    
    // like previous(), but keeps moving back until the trail goes to a node with isPainted() === true
    previousPainted: function() {
      var result = this.previous();
      while ( result && !result.isPainted() ) {
        result = result.previous();
      }
      return result;
    },
    
    // in the order of self-rendering
    next: function() {
      var arr = this.nodes.slice( 0 );
      
      var top = this.nodeFromTop( 0 );
      if ( top._children.length > 0 ) {
        // if we have children, return the first child
        arr.push( top._children[0] );
        return new Trail( arr );
      } else {
        // walk down and attempt to find the next parent
        var depth = this.nodes.length - 1;
        
        while ( depth > 0 ) {
          var node = this.nodes[depth];
          var parent = this.nodes[depth-1];
          
          arr.pop(); // take off the node so we can add the next sibling if it exists
          
          var index = _.indexOf( parent._children, node );
          if ( index !== parent._children.length - 1 ) {
            // there is another (later) sibling. use that!
            arr.push( parent._children[index+1] );
            return new Trail( arr );
          } else {
            depth--;
          }
        }
        
        // if we didn't reach a later sibling by now, it doesn't exist
        return null;
      }
    },
    
    // like next(), but keeps moving back until the trail goes to a node with isPainted() === true
    nextPainted: function() {
      var result = this.next();
      while ( result && !result.isPainted() ) {
        result = result.next();
      }
      return result;
    },
    
    // calls callback( trail ) for this trail, and each descendant trail
    eachTrailUnder: function( callback ) {
      new scenery.TrailPointer( this, true ).eachTrailBetween( new scenery.TrailPointer( this, false ), callback );
    },
    
    /*
     * Standard Java-style compare. -1 means this trail is before (under) the other trail, 0 means equal, and 1 means this trail is
     * after (on top of) the other trail.
     * A shorter subtrail will compare as -1.
     *
     * Assumes that the Trails are properly indexed. If not, please reindex them!
     *
     * Comparison is for the rendering order, so an ancestor is 'before' a descendant
     */
    compare: function( other ) {
      assert && assert( !this.isEmpty(), 'cannot compare with an empty trail' );
      assert && assert( !other.isEmpty(), 'cannot compare with an empty trail' );
      assert && assert( this.nodes[0] === other.nodes[0], 'for Trail comparison, trails must have the same root node' );
      assertExtra && assertExtra( this.areIndicesValid(), 'Trail.compare this.areIndicesValid() failed' );
      assertExtra && assertExtra( other.areIndicesValid(), 'Trail.compare other.areIndicesValid() failed' );
      
      var minNodeIndex = Math.min( this.indices.length, other.indices.length );
      for ( var i = 0; i < minNodeIndex; i++ ) {
        if ( this.indices[i] !== other.indices[i] ) {
          if ( this.indices[i] < other.indices[i] ) {
            return -1;
          } else {
            return 1;
          }
        }
      }
      
      // we scanned through and no nodes were different (one is a subtrail of the other)
      if ( this.nodes.length < other.nodes.length ) {
        return -1;
      } else if ( this.nodes.length > other.nodes.length ) {
        return 1;
      } else {
        return 0;
      }
    },
    
    localToGlobalPoint: function( point ) {
      return this.getTransform().transformPosition2( point );
    },
    
    localToGlobalBounds: function( bounds ) {
      return this.getTransform().transformBounds2( bounds );
    },
    
    globalToLocalPoint: function( point ) {
      return this.getTransform().inversePosition2( point );
    },
    
    globalToLocalBounds: function( bounds ) {
      return this.getTransform().inverseBounds2( bounds );
    },
    
    // concatenates the unique IDs of nodes in the trail, so that we can do id-based lookups
    getUniqueId: function() {
      // TODO: consider caching this if it is ever a bottleneck. it seems like it might be called in layer-refresh inner loops
      return _.map( this.nodes, function( node ) { return node.getId(); } ).join( '-' );
    },
    
    toString: function() {
      this.reindex();
      if ( !this.length ) {
        return 'Empty Trail';
      }
      return '[Trail ' + this.indices.join( '.' ) + ' ' + this.getUniqueId() + ']';
    }
  };
  
  // like eachTrailBetween, but only fires for painted trails
  Trail.eachPaintedTrailbetween = function( a, b, callback, excludeEndTrails, scene ) {
    Trail.eachTrailBetween( a, b, function( trail ) {
      if ( trail && trail.isPainted() ) {
        callback( trail );
      }
    }, excludeEndTrails, scene );
  };
  
  // global way of iterating across trails
  Trail.eachTrailBetween = function( a, b, callback, excludeEndTrails, scene ) {
    var aPointer = a ? new scenery.TrailPointer( a.copy(), true ) : new scenery.TrailPointer( new scenery.Trail( scene ), true );
    var bPointer = b ? new scenery.TrailPointer( b.copy(), true ) : new scenery.TrailPointer( new scenery.Trail( scene ), false );
    
    // if we are excluding endpoints, just bump the pointers towards each other by one step
    if ( excludeEndTrails ) {
      aPointer.nestedForwards();
      bPointer.nestedBackwards();
      
      // they were adjacent, so no callbacks will be executed
      if ( aPointer.compareNested( bPointer ) === 1 ) {
        return;
      }
    }
    
    aPointer.depthFirstUntil( bPointer, function( pointer ) {
      if ( pointer.isBefore ) {
        callback( pointer.trail );
      }
    }, false );
  };
  
  return Trail;
} );



// Copyright 2002-2012, University of Colorado

/*
 * A pointer is an abstraction that includes a mouse and touch points (and possibly keys).
 *
 * TODO: add state tracking (dragging/panning/etc.) to pointer for convenience
 * TODO: consider an 'active' flag?
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/input/Pointer',['require','ASSERT/assert','SCENERY/scenery'], function( require ) {
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Pointer = function() {
    this.listeners = [];
  };
  var Pointer = scenery.Pointer;
  
  Pointer.prototype = {
    constructor: Pointer,
    
    addInputListener: function( listener ) {
      assert && assert( !_.contains( this.listeners, listener ) );
      
      this.listeners.push( listener );
    },
    
    removeInputListener: function( listener ) {
      var index = _.indexOf( this.listeners, listener );
      assert && assert( index !== -1 );
      
      this.listeners.splice( index, 1 );
    }
  };
  
  return Pointer;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Tracks the mouse state
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/input/Mouse',['require','SCENERY/scenery','SCENERY/input/Pointer'], function( require ) {
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Pointer = require( 'SCENERY/input/Pointer' ); // inherits from Pointer
  
  scenery.Mouse = function() {
    Pointer.call( this );
    
    this.point = null;
    
    this.leftDown = false;
    this.middleDown = false;
    this.rightDown = false;
    
    this.isMouse = true;
    
    this.trail = null;
    
    this.type = 'mouse';
  };
  var Mouse = scenery.Mouse;
  
  Mouse.prototype = _.extend( {}, Pointer.prototype, {
    constructor: Mouse,
    
    down: function( point, event ) {
      this.point = point;
      switch( event.button ) {
        case 0: this.leftDown = true; break;
        case 1: this.middleDown = true; break;
        case 2: this.rightDown = true; break;
      }
    },
    
    up: function( point, event ) {
      this.point = point;
      switch( event.button ) {
        case 0: this.leftDown = false; break;
        case 1: this.middleDown = false; break;
        case 2: this.rightDown = false; break;
      }
    },
    
    move: function( point, event ) {
      this.point = point;
    },
    
    over: function( point, event ) {
      this.point = point;
    },
    
    out: function( point, event ) {
      // TODO: how to handle the mouse out-of-bounds
      this.point = null;
    }
  } );
  
  return Mouse;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Tracks a single touch point
 *
 * IE guidelines for Touch-friendly sites: http://blogs.msdn.com/b/ie/archive/2012/04/20/guidelines-for-building-touch-friendly-sites.aspx
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/input/Touch',['require','SCENERY/scenery','SCENERY/input/Pointer'], function( require ) {
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Pointer = require( 'SCENERY/input/Pointer' ); // extends Pointer
  
  scenery.Touch = function( id, point, event ) {
    Pointer.call( this );
    
    this.id = id;
    this.point = point;
    this.isTouch = true;
    this.trail = null;
    
    this.type = 'touch';
  };
  var Touch = scenery.Touch;
  
  Touch.prototype = _.extend( {}, Pointer.prototype, {
    constructor: Touch,
    
    move: function( point, event ) {
      this.point = point;
    },
    
    end: function( point, event ) {
      this.point = point;
    },
    
    cancel: function( point, event ) {
      this.point = point;
    }
  } );
  
  return Touch;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Tracks a stylus ('pen') or something with tilt and pressure information
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/input/Pen',['require','SCENERY/scenery','SCENERY/input/Pointer'], function( require ) {
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Pointer = require( 'SCENERY/input/Pointer' ); // extends Pointer
  
  scenery.Pen = function( id, point, event ) {
    Pointer.call( this );
    
    this.id = id;
    this.point = point;
    this.isPen = true;
    this.trail = null;
    
    this.type = 'pen';
  };
  var Pen = scenery.Pen;
  
  Pen.prototype = _.extend( {}, Pointer.prototype, {
    constructor: Pen,
    
    move: function( point, event ) {
      this.point = point;
    },
    
    end: function( point, event ) {
      this.point = point;
    },
    
    cancel: function( point, event ) {
      this.point = point;
    }
  } );
  
  return Pen;
} );

// Copyright 2002-2012, University of Colorado

/**
 * API for handling mouse / touch / keyboard events.
 *
 * A 'pointer' is an abstract way of describing either the mouse, a single touch point, or a key being pressed.
 * touch points and key presses go away after being released, whereas the mouse 'pointer' is persistent.
 *
 * Events will be called on listeners with a single event object. Supported event types are:
 * 'up', 'down', 'out', 'over', 'enter', 'exit', 'move', and 'cancel'. Scenery also supports more specific event
 * types that constrain the type of pointer, so 'mouse' + type, 'touch' + type and 'pen' + type will fire
 * on each listener before the generic event would be fined. E.g. for mouse movement, listener.mousemove will be
 * fired before listener.move.
 *
 * DOM Level 3 events spec: http://www.w3.org/TR/DOM-Level-3-Events/
 * Touch events spec: http://www.w3.org/TR/touch-events/
 * Pointer events spec draft: https://dvcs.w3.org/hg/pointerevents/raw-file/tip/pointerEvents.html
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/input/Input',['require','ASSERT/assert','SCENERY/scenery','SCENERY/util/Trail','SCENERY/input/Mouse','SCENERY/input/Touch','SCENERY/input/Pen','SCENERY/input/Event'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/input/Mouse' );
  require( 'SCENERY/input/Touch' );
  require( 'SCENERY/input/Pen' );
  require( 'SCENERY/input/Event' );
  
  scenery.Input = function( scene, listenerTarget ) {
    this.scene = scene;
    this.listenerTarget = listenerTarget;
    
    this.mouse = new scenery.Mouse();
    
    this.pointers = [ this.mouse ];
    
    this.listenerReferences = [];
  };
  var Input = scenery.Input;
  
  Input.prototype = {
    constructor: Input,
    
    addPointer: function( pointer ) {
      this.pointers.push( pointer );
    },
    
    removePointer: function( pointer ) {
      // sanity check version, will remove all instances
      for ( var i = this.pointers.length - 1; i >= 0; i-- ) {
        if ( this.pointers[i] === pointer ) {
          this.pointers.splice( i, 1 );
        }
      }
    },
    
    findTouchById: function( id ) {
      return _.find( this.pointers, function( pointer ) { return pointer.id === id; } );
    },
    
    mouseDown: function( point, event ) {
      this.mouse.down( point, event );
      this.downEvent( this.mouse, event );
    },
    
    mouseUp: function( point, event ) {
      this.mouse.up( point, event );
      this.upEvent( this.mouse, event );
    },
    
    mouseMove: function( point, event ) {
      this.mouse.move( point, event );
      this.moveEvent( this.mouse, event );
    },
    
    mouseOver: function( point, event ) {
      this.mouse.over( point, event );
      // TODO: how to handle mouse-over
    },
    
    mouseOut: function( point, event ) {
      this.mouse.out( point, event );
      // TODO: how to handle mouse-out
    },
    
    // called for each touch point
    touchStart: function( id, point, event ) {
      var touch = new scenery.Touch( id, point, event );
      this.addPointer( touch );
      this.downEvent( touch, event );
    },
    
    touchEnd: function( id, point, event ) {
      var touch = this.findTouchById( id );
      touch.end( point, event );
      this.removePointer( touch );
      this.upEvent( touch, event );
    },
    
    touchMove: function( id, point, event ) {
      var touch = this.findTouchById( id );
      touch.move( point, event );
      this.moveEvent( touch, event );
    },
    
    touchCancel: function( id, point, event ) {
      var touch = this.findTouchById( id );
      touch.cancel( point, event );
      this.removePointer( touch );
      this.cancelEvent( touch, event );
    },
    
    // called for each touch point
    penStart: function( id, point, event ) {
      var pen = new scenery.Pen( id, point, event );
      this.addPointer( pen );
      this.downEvent( pen, event );
    },
    
    penEnd: function( id, point, event ) {
      var pen = this.findTouchById( id );
      pen.end( point, event );
      this.removePointer( pen );
      this.upEvent( pen, event );
    },
    
    penMove: function( id, point, event ) {
      var pen = this.findTouchById( id );
      pen.move( point, event );
      this.moveEvent( pen, event );
    },
    
    penCancel: function( id, point, event ) {
      var pen = this.findTouchById( id );
      pen.cancel( point, event );
      this.removePointer( pen );
      this.cancelEvent( pen, event );
    },
    
    pointerDown: function( id, type, point, event ) {
      switch ( type ) {
        case 'mouse':
          this.mouseDown( point, event );
          break;
        case 'touch':
          this.touchStart( id, point, event );
          break;
        case 'pen':
          this.penStart( id, point, event );
          break;
        default:
          if ( console.log ) {
            console.log( 'Unknown pointer type: ' + type );
          }
      }
    },
    
    pointerUp: function( id, type, point, event ) {
      switch ( type ) {
        case 'mouse':
          this.mouseUp( point, event );
          break;
        case 'touch':
          this.touchEnd( id, point, event );
          break;
        case 'pen':
          this.penEnd( id, point, event );
          break;
        default:
          if ( console.log ) {
            console.log( 'Unknown pointer type: ' + type );
          }
      }
    },
    
    pointerCancel: function( id, type, point, event ) {
      switch ( type ) {
        case 'mouse':
          if ( console && console.log ) {
            console.log( 'WARNING: Pointer mouse cancel was received' );
          }
          break;
        case 'touch':
          this.touchCancel( id, point, event );
          break;
        case 'pen':
          this.penCancel( id, point, event );
          break;
        default:
          if ( console.log ) {
            console.log( 'Unknown pointer type: ' + type );
          }
      }
    },
    
    pointerMove: function( id, type, point, event ) {
      switch ( type ) {
        case 'mouse':
          this.mouseMove( point, event );
          break;
        case 'touch':
          this.touchMove( id, point, event );
          break;
        case 'pen':
          this.penMove( id, point, event );
          break;
        default:
          if ( console.log ) {
            console.log( 'Unknown pointer type: ' + type );
          }
      }
    },
    
    pointerOver: function( id, type, point, event ) {
      
    },
    
    pointerOut: function( id, type, point, event ) {
      
    },
    
    pointerEnter: function( id, type, point, event ) {
      
    },
    
    pointerLeave: function( id, type, point, event ) {
      
    },
    
    upEvent: function( pointer, event ) {
      var trail = this.scene.trailUnderPoint( pointer.point ) || new scenery.Trail( this.scene );
      
      this.dispatchEvent( trail, 'up', pointer, event, true );
      
      pointer.trail = trail;
    },
    
    downEvent: function( pointer, event ) {
      var trail = this.scene.trailUnderPoint( pointer.point ) || new scenery.Trail( this.scene );
      
      this.dispatchEvent( trail, 'down', pointer, event, true );
      
      pointer.trail = trail;
    },
    
    moveEvent: function( pointer, event ) {
      var trail = this.scene.trailUnderPoint( pointer.point ) || new scenery.Trail( this.scene );
      var oldTrail = pointer.trail || new scenery.Trail( this.scene );
      
      var lastNodeChanged = oldTrail.lastNode() !== trail.lastNode();
      
      var branchIndex;
      
      for ( branchIndex = 0; branchIndex < Math.min( trail.length, oldTrail.length ); branchIndex++ ) {
        if ( trail.nodes[branchIndex] !== oldTrail.nodes[branchIndex] ) {
          break;
        }
      }
      
      if ( lastNodeChanged ) {
        this.dispatchEvent( oldTrail, 'out', pointer, event, true );
      }
      
      // we want to approximately mimic http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order
      // TODO: if a node gets moved down 1 depth, it may see both an exit and enter?
      if ( oldTrail.length > branchIndex ) {
        for ( var oldIndex = branchIndex; oldIndex < oldTrail.length; oldIndex++ ) {
          this.dispatchEvent( oldTrail.slice( 0, oldIndex + 1 ), 'exit', pointer, event, false );
        }
      }
      if ( trail.length > branchIndex ) {
        for ( var newIndex = branchIndex; newIndex < trail.length; newIndex++ ) {
          this.dispatchEvent( trail.slice( 0, newIndex + 1 ), 'enter', pointer, event, false );
        }
      }
      
      if ( lastNodeChanged ) {
        this.dispatchEvent( trail, 'over', pointer, event, true );
      }
      
      // TODO: move the 'move' event to before the others, matching http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order ?
      this.dispatchEvent( trail, 'move', pointer, event, true );
      
      pointer.trail = trail;
    },
    
    cancelEvent: function( pointer, event ) {
      var trail = this.scene.trailUnderPoint( pointer.point );
      
      this.dispatchEvent( trail, 'cancel', pointer, event, true );
      
      pointer.trail = trail;
    },
    
    dispatchEvent: function( trail, type, pointer, event, bubbles ) {
      if ( !trail ) {
        try {
          throw new Error( 'falsy trail for dispatchEvent' );
        } catch ( e ) {
          console.log( e.stack );
          throw e;
        }
      }
      
      // TODO: is there a way to make this event immutable?
      var inputEvent = new scenery.Event( {
        trail: trail, // {Trail} path to the leaf-most node, ordered list, from root to leaf
        type: type, // {String} what event was triggered on the listener
        pointer: pointer, // {Pointer}
        domEvent: event, // raw DOM InputEvent (TouchEvent, PointerEvent, MouseEvent,...)
        currentTarget: null, // {Node} whatever node you attached the listener to, null when passed to a Pointer,
        target: trail.lastNode() // {Node} leaf-most node in trail
      } );
      
      // first run through the pointer's listeners to see if one of them will handle the event
      this.dispatchToPointer( type, pointer, inputEvent );
      
      // if not yet handled, run through the trail in order to see if one of them will handle the event
      // at the base of the trail should be the scene node, so the scene will be notified last
      this.dispatchToTargets( trail, pointer, type, inputEvent, bubbles );
      
      // TODO: better interactivity handling?
      if ( !trail.lastNode().interactive ) {
        event.preventDefault();
      }
    },
    
    // TODO: reduce code sharing between here and dispatchToTargets!
    dispatchToPointer: function( type, pointer, inputEvent ) {
      if ( inputEvent.aborted || inputEvent.handled ) {
        return;
      }
      
      var specificType = pointer.type + type; // e.g. mouseup, touchup, keyup
      
      var pointerListeners = pointer.listeners.slice( 0 ); // defensive copy
      for ( var i = 0; i < pointerListeners.length; i++ ) {
        var listener = pointerListeners[i];
        
        // if a listener returns true, don't handle any more
        var aborted = false;
        
        if ( !aborted && listener[specificType] ) {
          listener[specificType]( inputEvent );
          aborted = inputEvent.aborted;
        }
        if ( !aborted && listener[type] ) {
          listener[type]( inputEvent );
          aborted = inputEvent.aborted;
        }
        
        // bail out if the event is aborted, so no other listeners are triggered
        if ( aborted ) {
          return;
        }
      }
    },
    
    dispatchToTargets: function( trail, pointer, type, inputEvent, bubbles ) {
      if ( inputEvent.aborted || inputEvent.handled ) {
        return;
      }
      
      var specificType = pointer.type + type; // e.g. mouseup, touchup, keyup
      
      for ( var i = trail.length - 1; i >= 0; bubbles ? i-- : i = -1 ) {
        var target = trail.nodes[i];
        inputEvent.currentTarget = target;
        
        var listeners = target.getInputListeners();
        
        for ( var k = 0; k < listeners.length; k++ ) {
          var listener = listeners[k];
          
          // if a listener returns true, don't handle any more
          var aborted = false;
          
          if ( !aborted && listener[specificType] ) {
            listener[specificType]( inputEvent );
            aborted = inputEvent.aborted;
          }
          if ( !aborted && listener[type] ) {
            listener[type]( inputEvent );
            aborted = inputEvent.aborted;
          }
          
          // bail out if the event is aborted, so no other listeners are triggered
          if ( aborted ) {
            return;
          }
        }
        
        // if the input event was handled, don't follow the trail down another level
        if ( inputEvent.handled ) {
          return;
        }
      }
    },
    
    addListener: function( type, callback, useCapture ) {
      this.listenerTarget.addEventListener( type, callback, useCapture );
      this.listenerReferences.push( { type: type, callback: callback, useCapture: useCapture } );
    },
    
    diposeListeners: function() {
      var input = this;
      _.each( this.listenerReferences, function( ref ) {
        input.listenerTarget.removeEventListener( ref.type, ref.callback, ref.useCapture );
      } );
    }
  };
  
  return Input;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Tracks a single key-press
 *
 * TODO: general key-press implementation
 * TODO: consider separate handling for keys in general.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/input/Key',['require','SCENERY/scenery','SCENERY/input/Pointer'], function( require ) {
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Pointer = require( 'SCENERY/input/Pointer' ); // Inherits from Pointer
  
  scenery.Key = function( key, event ) {
    Pointer.call( this );
    
    this.key = key;
    this.isKey = true;
    this.trail = null;
    this.type = 'key';
  };
  var Key = scenery.Key;
  
  Key.prototype = _.extend( {}, Pointer.prototype, {
    constructor: Key
  } );
  
  return Key;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Basic dragging for a node.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/input/SimpleDragHandler',['require','ASSERT/assert','DOT/Matrix3','SCENERY/scenery'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Matrix3 = require( 'DOT/Matrix3' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  /*
   * Allowed options: {
   *    allowTouchSnag: false // allow touch swipes across an object to pick it up,
   *    start: null           // if non-null, called when a drag is started. start( event, trail )
   *    drag: null            // if non-null, called when the user moves something with a drag (not a start or end event).
   *                                                                         drag( event, trail )
   *    end: null             // if non-null, called when a drag is ended.   end( event, trail )
   *    translate:            // if this exists, translate( { delta: _, oldPosition: _, position: _ } ) will be called instead of directly translating the node
   * }
   */
  scenery.SimpleDragHandler = function( options ) {
    var handler = this;
    
    this.options = _.extend( {
      allowTouchSnag: false
    }, options );
    
    this.dragging              = false;     // whether a node is being dragged with this handler
    this.pointer                = null;      // the pointer doing the current dragging
    this.trail                 = null;      // stores the path to the node that is being dragged
    this.transform             = null;      // transform of the trail to our node (but not including our node, so we can prepend the deltas)
    this.node                  = null;      // the node that we are handling the drag for
    this.lastDragPoint         = null;      // the location of the drag at the previous event (so we can calculate a delta)
    this.startTransformMatrix  = null;      // the node's transform at the start of the drag, so we can reset on a touch cancel
    this.mouseButton           = undefined; // tracks which mouse button was pressed, so we can handle that specifically
    // TODO: consider mouse buttons as separate pointers?
    
    // if an ancestor is transformed, pin our node
    this.transformListener = {
      transform: function( args ) {
        if ( !handler.trail.isExtensionOf( args.trail, true ) ) {
          return;
        }
        
        var newMatrix = args.trail.getTransform().getMatrix();
        var oldMatrix = handler.transform.getMatrix();
        
        // if A was the trail's old transform, B is the trail's new transform, we need to apply (B^-1 A) to our node
        handler.node.prependMatrix( newMatrix.inverted().timesMatrix( oldMatrix ) );
        
        // store the new matrix so we can do deltas using it now
        handler.transform.set( newMatrix );
      }
    };
    
    // this listener gets added to the pointer when it starts dragging our node
    this.dragListener = {
      // mouse/touch up
      up: function( event ) {
        assert && assert( event.pointer === handler.pointer );
        if ( !event.pointer.isMouse || event.domEvent.button === handler.mouseButton ) {
          handler.endDrag( event );
        }
      },
      
      // touch cancel
      cancel: function( event ) {
        assert && assert( event.pointer === handler.pointer );
        handler.endDrag( event );
        
        // since it's a cancel event, go back!
        handler.node.setMatrix( handler.startTransformMatrix );
      },
      
      // mouse/touch move
      move: function( event ) {
        assert && assert( event.pointer === handler.pointer );
        
        var delta = handler.transform.inverseDelta2( handler.pointer.point.minus( handler.lastDragPoint ) );
        
        // move by the delta between the previous point, using the precomputed transform
        // prepend the translation on the node, so we can ignore whatever other transform state the node has
        if ( handler.options.translate ) {
          var translation = handler.node.getTransform().getMatrix().getTranslation();
          handler.options.translate( {
            delta: delta,
            oldPosition: translation,
            position: translation.plus( delta )
          } );
        } else {
          handler.node.translate( delta, true );
        }
        handler.lastDragPoint = handler.pointer.point;
        
        if ( handler.options.drag ) {
          // TODO: consider adding in a delta to the listener
          // TODO: add the position in to the listener
          handler.options.drag( event, handler.trail ); // new position (old position?) delta
        }
      }
    };
  };
  var SimpleDragHandler = scenery.SimpleDragHandler;
  
  SimpleDragHandler.prototype = {
    constructor: SimpleDragHandler,
    
    startDrag: function( event ) {
      // set a flag on the pointer so it won't pick up other nodes
      event.pointer.dragging = true;
      event.pointer.addInputListener( this.dragListener );
      event.trail.rootNode().addEventListener( this.transformListener );
      
      // set all of our persistent information
      this.dragging = true;
      this.pointer = event.pointer;
      this.trail = event.trail.subtrailTo( event.currentTarget, true );
      this.transform = this.trail.getTransform();
      this.node = event.currentTarget;
      this.lastDragPoint = event.pointer.point;
      this.startTransformMatrix = event.currentTarget.getMatrix();
      this.mouseButton = event.domEvent.button; // should be undefined for touch events
      
      if ( this.options.start ) {
        this.options.start( event, this.trail );
      }
    },
    
    endDrag: function( event ) {
      this.pointer.dragging = false;
      this.pointer.removeInputListener( this.dragListener );
      this.trail.rootNode().removeEventListener( this.transformListener );
      this.dragging = false;
      
      if ( this.options.end ) {
        this.options.end( event, this.trail );
      }
    },
    
    tryToSnag: function( event ) {
      // only start dragging if the pointer isn't dragging anything, we aren't being dragged, and if it's a mouse it's button is down
      if ( !this.dragging && !event.pointer.dragging ) {
        this.startDrag( event );
      }
    },
    
    /*---------------------------------------------------------------------------*
    * events called from the node input listener
    *----------------------------------------------------------------------------*/
    
    // mouse/touch down on this node
    down: function( event ) {
      this.tryToSnag( event );
    },
    
    // touch enters this node
    touchenter: function( event ) {
      // allow touches to start a drag by moving "over" this node
      if ( this.options.allowTouchSnag ) {
        this.tryToSnag( event );
      }
    }
  };
  
  return SimpleDragHandler;
} );



// Copyright 2002-2012, University of Colorado

/**
 * The main 'kite' namespace object for the exported (non-Require.js) API. Used internally
 * since it prevents Require.js issues with circular dependencies.
 *
 * The returned kite object namespace may be incomplete if not all modules are listed as
 * dependencies. Please use the 'main' module for that purpose if all of Kite is desired.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/kite',['require'], function( require ) {
  // will be filled in by other modules
  return {};
} );

// Copyright 2002-2012, University of Colorado

/**
 * A segment represents a specific curve with a start and end.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/segments/Segment',['require','ASSERT/assert','KITE/kite'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );

  var kite = require( 'KITE/kite' );
  
  kite.Segment = {
    /*
     * Will contain (for segments):
     * properties:
     * start        - start point of this segment
     * end          - end point of this segment
     * startTangent - the tangent vector (normalized) to the segment at the start, pointing in the direction of motion (from start to end)
     * endTangent   - the tangent vector (normalized) to the segment at the end, pointing in the direction of motion (from start to end)
     * bounds       - the bounding box for the segment
     *
     * methods:
     * positionAt( t )          - returns the position parametrically, with 0 <= t <= 1. this does NOT guarantee a constant magnitude tangent... don't feel like adding elliptical functions yet!
     * tangentAt( t )           - returns the non-normalized tangent (dx/dt, dy/dt) parametrically, with 0 <= t <= 1.
     * curvatureAt( t )         - returns the signed curvature (positive for visual clockwise - mathematical counterclockwise)
     * toPieces                 - returns an array of pieces that are equivalent to this segment, assuming start points are preserved
     *                              TODO: is toPieces that valuable? it doesn't seem to have a strict guarantee on checking what the last segment did right now
     * getSVGPathFragment       - returns a string containing the SVG path. assumes that the start point is already provided, so anything that calls this needs to put the M calls first
     * strokeLeft( lineWidth )  - returns an array of pieces that will draw an offset curve on the logical left side
     * strokeRight( lineWidth ) - returns an array of pieces that will draw an offset curve on the logical right side
     * intersectsBounds         - whether this segment intersects the specified bounding box (not just the segment's bounding box, but the actual segment)
     * windingIntersection      - returns the winding number for intersection with a ray
     */
  };
  var Segment = kite.Segment;
  
  return Segment;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Represents an immutable higher-level command for Shape, and generally mimics the Canvas drawing api.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/pieces/Piece',['require','ASSERT/assert','KITE/kite'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );
  
  var kite = require( 'KITE/kite' );
  
  kite.Piece = {
    /*
     * Will contain (for pieces):
     * methods:
     * writeToContext( context ) - Executes this drawing command directly on a Canvas context
     * transformed( matrix )     - Returns a transformed copy of this piece
     * applyPiece( shape )       - Applies this piece to a shape, essentially internally executing the Canvas api and creating subpaths and segments.
     *                             This is necessary, since pieces like Rect can actually contain many more than one segment, and drawing pieces depends
     *                             on context / subpath state.
     */
  };
  var Piece = kite.Piece;
  
  return Piece;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Linear segment
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/segments/Line',['require','ASSERT/assert','KITE/kite','DOT/Bounds2','DOT/Util','KITE/segments/Segment','KITE/pieces/Piece'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );

  var kite = require( 'KITE/kite' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var lineLineIntersection = require( 'DOT/Util' ).lineLineIntersection;
  
  var Segment = require( 'KITE/segments/Segment' );
  var Piece = require( 'KITE/pieces/Piece' );

  Segment.Line = function( start, end ) {
    this.start = start;
    this.end = end;
    
    if ( start.equals( end, 0 ) ) {
      this.invalid = true;
      return;
    }
    
    this.startTangent = end.minus( start ).normalized();
    this.endTangent = this.startTangent;
    
    // acceleration for intersection
    this.bounds = Bounds2.NOTHING.withPoint( start ).withPoint( end );
  };
  Segment.Line.prototype = {
    constructor: Segment.Line,
    
    positionAt: function( t ) {
      return this.start.plus( this.end.minus( this.start ).times( t ) );
    },
    
    tangentAt: function( t ) {
      // tangent always the same, just use the start tanget
      return this.startTangent;
    },
    
    curvatureAt: function( t ) {
      return 0; // no curvature on a straight line segment
    },
    
    toPieces: function() {
      return [ new Piece.LineTo( this.end ) ];
    },
    
    getSVGPathFragment: function() {
      return 'L ' + this.end.x + ' ' + this.end.y;
    },
    
    strokeLeft: function( lineWidth ) {
      return [ new Piece.LineTo( this.end.plus( this.endTangent.perpendicular().negated().times( lineWidth / 2 ) ) ) ];
    },
    
    strokeRight: function( lineWidth ) {
      return [ new Piece.LineTo( this.start.plus( this.startTangent.perpendicular().times( lineWidth / 2 ) ) ) ];
    },
    
    intersectsBounds: function( bounds ) {
      throw new Error( 'Segment.Line.intersectsBounds unimplemented' ); // TODO: implement
    },
    
    intersection: function( ray ) {
      var result = [];
      
      var start = this.start;
      var end = this.end;
      
      var intersection = lineLineIntersection( start, end, ray.pos, ray.pos.plus( ray.dir ) );
      
      if ( !isFinite( intersection.x ) || !isFinite( intersection.y ) ) {
        // lines must be parallel
        return result;
      }
      
      // check to make sure our point is in our line segment (specifically, in the bounds (start,end], not including the start point so we don't double-count intersections)
      if ( start.x !== end.x && ( start.x > end.x ? ( intersection.x >= start.x || intersection.x < end.x ) : ( intersection.x <= start.x || intersection.x > end.x ) ) ) {
        return result;
      }
      if ( start.y !== end.y && ( start.y > end.y ? ( intersection.y >= start.y || intersection.y < end.y ) : ( intersection.y <= start.y || intersection.y > end.y ) ) ) {
        return result;
      }
      
      // make sure the intersection is not behind the ray
      var t = intersection.minus( ray.pos ).dot( ray.dir );
      if ( t < 0 ) {
        return result;
      }
      
      // return the proper winding direction depending on what way our line intersection is "pointed"
      var diff = end.minus( start );
      var perp = diff.perpendicular();
      result.push( {
        distance: t,
        point: ray.pointAtDistance( t ),
        normal: perp.dot( ray.dir ) > 0 ? perp.negated() : perp,
        wind: ray.dir.perpendicular().dot( diff ) < 0 ? 1 : -1
      } );
      return result;
    },
    
    // returns the resultant winding number of this ray intersecting this segment.
    windingIntersection: function( ray ) {
      var hits = this.intersection( ray );
      if ( hits.length ) {
        return hits[0].wind;
      } else {
        return 0;
      }
    }
  };
  
  return Segment.Line;
} );

// Copyright 2002-2012, University of Colorado

/**
 * A Canvas-style stateful (mutable) subpath, which tracks segments in addition to the points.
 *
 * See http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#concept-path
 * for the path / subpath Canvas concept.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/util/Subpath',['require','ASSERT/assert','KITE/kite','KITE/segments/Line'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );
  
  var kite = require( 'KITE/kite' );
  
  require( 'KITE/segments/Line' );
  
  kite.Subpath = function() {
    this.points = [];
    this.segments = [];
    this.closed = false;
  };
  var Subpath = kite.Subpath;
  Subpath.prototype = {
    addPoint: function( point ) {
      this.points.push( point );
    },
    
    addSegment: function( segment ) {
      if ( !segment.invalid ) {
        assert && assert( segment.start.isFinite(), 'Segment start is infinite' );
        assert && assert( segment.end.isFinite(), 'Segment end is infinite' );
        assert && assert( segment.startTangent.isFinite(), 'Segment startTangent is infinite' );
        assert && assert( segment.endTangent.isFinite(), 'Segment endTangent is infinite' );
        assert && assert( segment.bounds.isEmpty() || segment.bounds.isFinite(), 'Segment bounds is infinite and non-empty' );
        this.segments.push( segment );
      }
    },
    
    close: function() {
      this.closed = true;
    },
    
    getLength: function() {
      return this.points.length;
    },
    
    getFirstPoint: function() {
      return _.first( this.points );
    },
    
    getLastPoint: function() {
      return _.last( this.points );
    },
    
    getFirstSegment: function() {
      return _.first( this.segments );
    },
    
    getLastSegment: function() {
      return _.last( this.segments );
    },
    
    isDrawable: function() {
      return this.segments.length > 0;
    },
    
    isClosed: function() {
      return this.closed;
    },
    
    hasClosingSegment: function() {
      return !this.getFirstPoint().equalsEpsilon( this.getLastPoint(), 0.000000001 );
    },
    
    getClosingSegment: function() {
      assert && assert( this.hasClosingSegment(), 'Implicit closing segment unnecessary on a fully closed path' );
      return new kite.Segment.Line( this.getLastPoint(), this.getFirstPoint() );
    }
  };
  
  return kite.Subpath;
} );

//jshint -W018
// Copyright 2002-2012, University of Colorado

/**
 * Styles needed to determine a stroked line shape.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/util/LineStyles',['require','ASSERT/assert','KITE/kite'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );
  
  var kite = require( 'KITE/kite' );
  
  kite.LineStyles = function( args ) {
    if ( args === undefined ) {
      args = {};
    }
    this.lineWidth = args.lineWidth !== undefined ? args.lineWidth : 1;
    this.lineCap = args.lineCap !== undefined ? args.lineCap : 'butt'; // butt, round, square
    this.lineJoin = args.lineJoin !== undefined ? args.lineJoin : 'miter'; // miter, round, bevel
    this.lineDash = args.lineDash !== undefined ? args.lineDash : null; // null is default, otherwise an array of numbers
    this.lineDashOffset = args.lineDashOffset !== undefined ? args.lineDashOffset : 0; // 0 default, any number
    this.miterLimit = args.miterLimit !== undefined ? args.miterLimit : 10; // see https://svgwg.org/svg2-draft/painting.html for miterLimit computations
  };
  var LineStyles = kite.LineStyles;
  LineStyles.prototype = {
    constructor: LineStyles,
    
    equals: function( other ) {
      var typical = this.lineWidth === other.lineWidth &&
                    this.lineCap === other.lineCap &&
                    this.lineJoin === other.lineJoin &&
                    this.miterLimit === other.miterLimit &&
                    this.lineDashOffset === other.lineDashOffset;
      if ( !typical ) {
        return false;
      }
      
      // now we need to compare the line dashes
      /* jshint -W018 */
      //jshint -W018
      if ( !this.lineDash !== !other.lineDash ) {
        // one is defined, the other is not
        return false;
      }
      
      if ( this.lineDash ) {
        if ( this.lineDash.length !== other.lineDash.length ) {
          return false;
        }
        for ( var i = 0; i < this.lineDash.length; i++ ) {
          if ( this.lineDash[i] !== other.lineDash[i] ) {
            return false;
          }
        }
        return true;
      } else {
        // both have no line dash, so they are equal
        return true;
      }
    }
  };
  
  return kite.LineStyles;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Elliptical arc segment
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/segments/EllipticalArc',['require','ASSERT/assert','KITE/kite','DOT/Vector2','DOT/Bounds2','DOT/Matrix3','DOT/Transform3','DOT/Util','KITE/segments/Segment','KITE/pieces/Piece','KITE/util/Subpath'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );

  var kite = require( 'KITE/kite' );
  
  var Vector2 = require( 'DOT/Vector2' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Transform3 = require( 'DOT/Transform3' );
  var toDegrees = require( 'DOT/Util' ).toDegrees;

  var Segment = require( 'KITE/segments/Segment' );
  var Piece = require( 'KITE/pieces/Piece' );
  require( 'KITE/util/Subpath' );

  // TODO: notes at http://www.w3.org/TR/SVG/implnote.html#PathElementImplementationNotes
  // Canvas notes at http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-ellipse
  Segment.EllipticalArc = function( center, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) {
    this.center = center;
    this.radiusX = radiusX;
    this.radiusY = radiusY;
    this.rotation = rotation;
    this.startAngle = startAngle;
    this.endAngle = endAngle;
    this.anticlockwise = anticlockwise;
    
    this.unitTransform = Segment.EllipticalArc.computeUnitTransform( center, radiusX, radiusY, rotation );
    
    this.start = this.positionAtAngle( startAngle );
    this.end = this.positionAtAngle( endAngle );
    this.startTangent = this.tangentAtAngle( startAngle ).normalized();
    this.endTangent = this.tangentAtAngle( endAngle ).normalized();
    
    if ( radiusX === 0 || radiusY === 0 || startAngle === endAngle ) {
      this.invalid = true;
      return;
    }
    
    if ( radiusX < radiusY ) {
      // TODO: check this
      throw new Error( 'Not verified to work if radiusX < radiusY' );
    }
    
    // constraints shared with Segment.Arc
    assert && assert( !( ( !anticlockwise && endAngle - startAngle <= -Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle <= -Math.PI * 2 ) ), 'Not handling elliptical arcs with start/end angles that show differences in-between browser handling' );
    assert && assert( !( ( !anticlockwise && endAngle - startAngle > Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle > Math.PI * 2 ) ), 'Not handling elliptical arcs with start/end angles that show differences in-between browser handling' );
    
    var isFullPerimeter = ( !anticlockwise && endAngle - startAngle >= Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle >= Math.PI * 2 );
    
    // compute an angle difference that represents how "much" of the circle our arc covers
    this.angleDifference = this.anticlockwise ? this.startAngle - this.endAngle : this.endAngle - this.startAngle;
    if ( this.angleDifference < 0 ) {
      this.angleDifference += Math.PI * 2;
    }
    assert && assert( this.angleDifference >= 0 ); // now it should always be zero or positive
    
    // a unit arg segment that we can map to our ellipse. useful for hit testing and such.
    this.unitArcSegment = new Segment.Arc( Vector2.ZERO, 1, startAngle, endAngle, anticlockwise );
    
    this.bounds = Bounds2.NOTHING;
    this.bounds = this.bounds.withPoint( this.start );
    this.bounds = this.bounds.withPoint( this.end );
    
    // for bounds computations
    var that = this;
    function boundsAtAngle( angle ) {
      if ( that.containsAngle( angle ) ) {
        // the boundary point is in the arc
        that.bounds = that.bounds.withPoint( that.positionAtAngle( angle ) );
      }
    }
    
    // if the angles are different, check extrema points
    if ( startAngle !== endAngle ) {
      // solve the mapping from the unit circle, find locations where a coordinate of the gradient is zero.
      // we find one extrema point for both x and y, since the other two are just rotated by pi from them.
      var xAngle = Math.atan( -( radiusY / radiusX ) * Math.tan( rotation ) );
      var yAngle = Math.atan( ( radiusY / radiusX ) / Math.tan( rotation ) );
      
      // check all of the extrema points
      boundsAtAngle( xAngle );
      boundsAtAngle( xAngle + Math.PI );
      boundsAtAngle( yAngle );
      boundsAtAngle( yAngle + Math.PI );
    }
  };
  Segment.EllipticalArc.prototype = {
    constructor: Segment.EllipticalArc,
    
    angleAt: function( t ) {
      if ( this.anticlockwise ) {
        // angle is 'decreasing'
        // -2pi <= end - start < 2pi
        if ( this.startAngle > this.endAngle ) {
          return this.startAngle + ( this.endAngle - this.startAngle ) * t;
        } else if ( this.startAngle < this.endAngle ) {
          return this.startAngle + ( -Math.PI * 2 + this.endAngle - this.startAngle ) * t;
        } else {
          // equal
          return this.startAngle;
        }
      } else {
        // angle is 'increasing'
        // -2pi < end - start <= 2pi
        if ( this.startAngle < this.endAngle ) {
          return this.startAngle + ( this.endAngle - this.startAngle ) * t;
        } else if ( this.startAngle > this.endAngle ) {
          return this.startAngle + ( Math.PI * 2 + this.endAngle - this.startAngle ) * t;
        } else {
          // equal
          return this.startAngle;
        }
      }
    },
    
    positionAt: function( t ) {
      return this.positionAtAngle( this.angleAt( t ) );
    },
    
    tangentAt: function( t ) {
      return this.tangentAtAngle( this.angleAt( t ) );
    },
    
    curvatureAt: function( t ) {
      // see http://mathworld.wolfram.com/Ellipse.html (59)
      var angle = this.angleAt( t );
      var aq = this.radiusX * Math.sin( angle );
      var bq = this.radiusY * Math.cos( angle );
      var denominator = Math.pow( bq * bq + aq * aq, 3/2 );
      return ( this.anticlockwise ? -1 : 1 ) * this.radiusX * this.radiusY / denominator;
    },
    
    positionAtAngle: function( angle ) {
      return this.unitTransform.transformPosition2( Vector2.createPolar( 1, angle ) );
    },
    
    tangentAtAngle: function( angle ) {
      var normal = this.unitTransform.transformNormal2( Vector2.createPolar( 1, angle ) );
      
      return this.anticlockwise ? normal.perpendicular() : normal.perpendicular().negated();
    },
    
    // TODO: refactor? exact same as Segment.Arc
    containsAngle: function( angle ) {
      // transform the angle into the appropriate coordinate form
      // TODO: check anticlockwise version!
      var normalizedAngle = this.anticlockwise ? angle - this.endAngle : angle - this.startAngle;
      
      // get the angle between 0 and 2pi
      var positiveMinAngle = normalizedAngle % ( Math.PI * 2 );
      // check this because modular arithmetic with negative numbers reveal a negative number
      if ( positiveMinAngle < 0 ) {
        positiveMinAngle += Math.PI * 2;
      }
      
      return positiveMinAngle <= this.angleDifference;
    },
    
    toPieces: function() {
      return [ new Piece.EllipticalArc( this.center, this.radiusX, this.radiusY, this.rotation, this.startAngle, this.endAngle, this.anticlockwise ) ];
    },
    
    // discretizes the elliptical arc and returns an offset curve as a list of lineTos
    offsetTo: function( r, reverse ) {
      // how many segments to create (possibly make this more adaptive?)
      var quantity = 32;
      
      var result = [];
      for ( var i = 1; i < quantity; i++ ) {
        var ratio = i / ( quantity - 1 );
        if ( reverse ) {
          ratio = 1 - ratio;
        }
        var angle = this.angleAt( ratio );
        
        var point = this.positionAtAngle( angle ).plus( this.tangentAtAngle( angle ).perpendicular().normalized().times( r ) );
        result.push( new Piece.LineTo( point ) );
      }
      
      return result;
    },
    
    getSVGPathFragment: function() {
      // see http://www.w3.org/TR/SVG/paths.html#PathDataEllipticalArcCommands for more info
      // rx ry x-axis-rotation large-arc-flag sweep-flag x y
      var epsilon = 0.01; // allow some leeway to render things as 'almost circles'
      var sweepFlag = this.anticlockwise ? '0' : '1';
      var largeArcFlag;
      var degreesRotation = toDegrees( this.rotation ); // bleh, degrees?
      if ( this.angleDifference < Math.PI * 2 - epsilon ) {
        largeArcFlag = this.angleDifference < Math.PI ? '0' : '1';
        return 'A ' + this.radiusX + ' ' + this.radiusY + ' ' + degreesRotation + ' ' + largeArcFlag + ' ' + sweepFlag + ' ' + this.end.x + ' ' + this.end.y;
      } else {
        // ellipse (or almost-ellipse) case needs to be handled differently
        // since SVG will not be able to draw (or know how to draw) the correct circle if we just have a start and end, we need to split it into two circular arcs
        
        // get the angle that is between and opposite of both of the points
        var splitOppositeAngle = ( this.startAngle + this.endAngle ) / 2; // this _should_ work for the modular case?
        var splitPoint = this.positionAtAngle( splitOppositeAngle );
        
        largeArcFlag = '0'; // since we split it in 2, it's always the small arc
        
        var firstArc = 'A ' + this.radiusX + ' ' + this.radiusY + ' ' + degreesRotation + ' ' + largeArcFlag + ' ' + sweepFlag + ' ' + splitPoint.x + ' ' + splitPoint.y;
        var secondArc = 'A ' + this.radiusX + ' ' + this.radiusY + ' ' + degreesRotation + ' ' + largeArcFlag + ' ' + sweepFlag + ' ' + this.end.x + ' ' + this.end.y;
        
        return firstArc + ' ' + secondArc;
      }
    },
    
    strokeLeft: function( lineWidth ) {
      return this.offsetTo( -lineWidth / 2, false );
    },
    
    strokeRight: function( lineWidth ) {
      return this.offsetTo( lineWidth / 2, true );
    },
    
    intersectsBounds: function( bounds ) {
      throw new Error( 'Segment.EllipticalArc.intersectsBounds unimplemented' );
    },
    
    intersection: function( ray ) {
      // be lazy. transform it into the space of a non-elliptical arc.
      var unitTransform = this.unitTransform;
      var rayInUnitCircleSpace = unitTransform.inverseRay2( ray );
      var hits = this.unitArcSegment.intersection( rayInUnitCircleSpace );
      
      return _.map( hits, function( hit ) {
        var transformedPoint = unitTransform.transformPosition2( hit.point );
        return {
          distance: ray.pos.distance( transformedPoint ),
          point: transformedPoint,
          normal: unitTransform.inverseNormal2( hit.normal ),
          wind: hit.wind
        };
      } );
    },
    
    // returns the resultant winding number of this ray intersecting this segment.
    windingIntersection: function( ray ) {
      // be lazy. transform it into the space of a non-elliptical arc.
      var rayInUnitCircleSpace = this.unitTransform.inverseRay2( ray );
      return this.unitArcSegment.windingIntersection( rayInUnitCircleSpace );
    }
  };
  
  // adapted from http://www.w3.org/TR/SVG/implnote.html#PathElementImplementationNotes
  // transforms the unit circle onto our ellipse
  Segment.EllipticalArc.computeUnitTransform = function( center, radiusX, radiusY, rotation ) {
    return new Transform3( Matrix3.translation( center.x, center.y ) // TODO: convert to Matrix3.translation( this.center) when available
                                  .timesMatrix( Matrix3.rotation2( rotation ) )
                                  .timesMatrix( Matrix3.scaling( radiusX, radiusY ) ) );
  };
  
  return Segment.EllipticalArc;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Elliptical arc piece
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/pieces/EllipticalArc',['require','ASSERT/assert','ASSERT/assert','KITE/kite','DOT/Vector2','DOT/Bounds2','DOT/Ray2','DOT/Matrix3','DOT/Transform3','KITE/pieces/Piece','KITE/segments/EllipticalArc','KITE/segments/Line','KITE/util/Subpath'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );
  var assertExtra = require( 'ASSERT/assert' )( 'kite.extra', true );
  
  var kite = require( 'KITE/kite' );
  
  var Vector2 = require( 'DOT/Vector2' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Ray2 = require( 'DOT/Ray2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Transform3 = require( 'DOT/Transform3' );

  var Piece = require( 'KITE/pieces/Piece' );
  require( 'KITE/segments/EllipticalArc' );
  require( 'KITE/segments/Line' );
  require( 'KITE/util/Subpath' );
  
  Piece.EllipticalArc = function( center, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) {
    if ( radiusX < 0 ) {
      // support this case since we might actually need to handle it inside of strokes?
      radiusX = -radiusX;
      startAngle = Math.PI - startAngle;
      endAngle = Math.PI - endAngle;
      anticlockwise = !anticlockwise;
    }
    if ( radiusY < 0 ) {
      // support this case since we might actually need to handle it inside of strokes?
      radiusY = -radiusY;
      startAngle = -startAngle;
      endAngle = -endAngle;
      anticlockwise = !anticlockwise;
    }
    if ( radiusX < radiusY ) {
      // swap radiusX and radiusY internally for consistent Canvas / SVG output
      rotation += Math.PI / 2;
      startAngle -= Math.PI / 2;
      endAngle -= Math.PI / 2;
      
      // swap radiusX and radiusY
      var tmpR = radiusX;
      radiusX = radiusY;
      radiusY = tmpR;
    }
    this.center = center;
    this.radiusX = radiusX;
    this.radiusY = radiusY;
    this.rotation = rotation;
    this.startAngle = startAngle;
    this.endAngle = endAngle;
    this.anticlockwise = anticlockwise;
    
    this.unitTransform = kite.Segment.EllipticalArc.computeUnitTransform( center, radiusX, radiusY, rotation );
  };
  Piece.EllipticalArc.prototype = {
    constructor: Piece.EllipticalArc,
    
    writeToContext: function( context ) {
      if ( context.ellipse ) {
        context.ellipse( this.center.x, this.center.y, this.radiusX, this.radiusY, this.rotation, this.startAngle, this.endAngle, this.anticlockwise );
      } else {
        // fake the ellipse call by using transforms
        this.unitTransform.getMatrix().canvasAppendTransform( context );
        context.arc( 0, 0, 1, this.startAngle, this.endAngle, this.anticlockwise );
        this.unitTransform.getInverse().canvasAppendTransform( context );
      }
    },
    
    // TODO: test various transform types, especially rotations, scaling, shears, etc.
    transformed: function( matrix ) {
      var transformedSemiMajorAxis = matrix.timesVector2( Vector2.createPolar( this.radiusX, this.rotation ) ).minus( matrix.timesVector2( Vector2.ZERO ) );
      var transformedSemiMinorAxis = matrix.timesVector2( Vector2.createPolar( this.radiusY, this.rotation + Math.PI / 2 ) ).minus( matrix.timesVector2( Vector2.ZERO ) );
      var rotation = transformedSemiMajorAxis.angle();
      var radiusX = transformedSemiMajorAxis.magnitude();
      var radiusY = transformedSemiMinorAxis.magnitude();
      
      var reflected = matrix.getDeterminant() < 0;
      
      // reverse the 'clockwiseness' if our transform includes a reflection
      // TODO: check reflections. swapping angle signs should fix clockwiseness
      var anticlockwise = reflected ? !this.anticlockwise : this.anticlockwise;
      var startAngle = reflected ? -this.startAngle : this.startAngle;
      var endAngle = reflected ? -this.endAngle : this.endAngle;
      
      return [new Piece.EllipticalArc( matrix.timesVector2( this.center ), radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise )];
    },
    
    applyPiece: function( shape ) {
      // see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-arc
      
      var ellipticalArc = new kite.Segment.EllipticalArc( this.center, this.radiusX, this.radiusY, this.rotation, this.startAngle, this.endAngle, this.anticlockwise );
      
      // we are assuming that the normal conditions were already met (or exceptioned out) so that these actually work with canvas
      var startPoint = ellipticalArc.start;
      var endPoint = ellipticalArc.end;
      
      // if there is already a point on the subpath, and it is different than our starting point, draw a line between them
      if ( shape.hasSubpaths() && shape.getLastSubpath().getLength() > 0 && !startPoint.equals( shape.getLastSubpath().getLastPoint(), 0 ) ) {
        shape.getLastSubpath().addSegment( new kite.Segment.Line( shape.getLastSubpath().getLastPoint(), startPoint ) );
      }
      
      if ( !shape.hasSubpaths() ) {
        shape.addSubpath( new kite.Subpath() );
      }
      
      shape.getLastSubpath().addSegment( ellipticalArc );
      
      // technically the Canvas spec says to add the start point, so we do this even though it is probably completely unnecessary (there is no conditional)
      shape.getLastSubpath().addPoint( startPoint );
      shape.getLastSubpath().addPoint( endPoint );
      
      // and update the bounds
      if ( !ellipticalArc.invalid ) {
        shape.bounds = shape.bounds.union( ellipticalArc.bounds );
      }
    },
    
    toString: function() {
      return 'ellipticalArc( ' + this.center.x + ', ' + this.center.y + ', ' + this.radiusX + ', ' + this.radiusY + ', ' + this.rotation + ', ' + this.startAngle + ', ' + this.endAngle + ', ' + this.anticlockwise + ' )';
    }
  };
  
  return Piece.EllipticalArc;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Arc segment
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/segments/Arc',['require','ASSERT/assert','KITE/kite','DOT/Vector2','DOT/Bounds2','KITE/segments/Segment','KITE/pieces/Piece'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );

  var kite = require( 'KITE/kite' );
  
  var Vector2 = require( 'DOT/Vector2' );
  var Bounds2 = require( 'DOT/Bounds2' );

  var Segment = require( 'KITE/segments/Segment' );
  var Piece = require( 'KITE/pieces/Piece' );

  Segment.Arc = function( center, radius, startAngle, endAngle, anticlockwise ) {
    this.center = center;
    this.radius = radius;
    this.startAngle = startAngle;
    this.endAngle = endAngle;
    this.anticlockwise = anticlockwise;
    
    this.start = this.positionAtAngle( startAngle );
    this.end = this.positionAtAngle( endAngle );
    this.startTangent = this.tangentAtAngle( startAngle );
    this.endTangent = this.tangentAtAngle( endAngle );
    
    if ( radius <= 0 || startAngle === endAngle ) {
      this.invalid = true;
      return;
    }
    // constraints
    assert && assert( !( ( !anticlockwise && endAngle - startAngle <= -Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle <= -Math.PI * 2 ) ), 'Not handling arcs with start/end angles that show differences in-between browser handling' );
    assert && assert( !( ( !anticlockwise && endAngle - startAngle > Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle > Math.PI * 2 ) ), 'Not handling arcs with start/end angles that show differences in-between browser handling' );
    
    var isFullPerimeter = ( !anticlockwise && endAngle - startAngle >= Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle >= Math.PI * 2 );
    
    // compute an angle difference that represents how "much" of the circle our arc covers
    this.angleDifference = this.anticlockwise ? this.startAngle - this.endAngle : this.endAngle - this.startAngle;
    if ( this.angleDifference < 0 ) {
      this.angleDifference += Math.PI * 2;
    }
    assert && assert( this.angleDifference >= 0 ); // now it should always be zero or positive
    
    // acceleration for intersection
    this.bounds = Bounds2.NOTHING;
    this.bounds = this.bounds.withPoint( this.start );
    this.bounds = this.bounds.withPoint( this.end );
    
    // for bounds computations
    var that = this;
    function boundsAtAngle( angle ) {
      if ( that.containsAngle( angle ) ) {
        // the boundary point is in the arc
        that.bounds = that.bounds.withPoint( center.plus( Vector2.createPolar( radius, angle ) ) );
      }
    }
    
    // if the angles are different, check extrema points
    if ( startAngle !== endAngle ) {
      // check all of the extrema points
      boundsAtAngle( 0 );
      boundsAtAngle( Math.PI / 2 );
      boundsAtAngle( Math.PI );
      boundsAtAngle( 3 * Math.PI / 2 );
    }
  };
  Segment.Arc.prototype = {
    constructor: Segment.Arc,
    
    angleAt: function( t ) {
      if ( this.anticlockwise ) {
        // angle is 'decreasing'
        // -2pi <= end - start < 2pi
        if ( this.startAngle > this.endAngle ) {
          return this.startAngle + ( this.endAngle - this.startAngle ) * t;
        } else if ( this.startAngle < this.endAngle ) {
          return this.startAngle + ( -Math.PI * 2 + this.endAngle - this.startAngle ) * t;
        } else {
          // equal
          return this.startAngle;
        }
      } else {
        // angle is 'increasing'
        // -2pi < end - start <= 2pi
        if ( this.startAngle < this.endAngle ) {
          return this.startAngle + ( this.endAngle - this.startAngle ) * t;
        } else if ( this.startAngle > this.endAngle ) {
          return this.startAngle + ( Math.PI * 2 + this.endAngle - this.startAngle ) * t;
        } else {
          // equal
          return this.startAngle;
        }
      }
    },
    
    positionAt: function( t ) {
      return this.positionAtAngle( this.angleAt( t ) );
    },
    
    tangentAt: function( t ) {
      return this.tangentAtAngle( this.angleAt( t ) );
    },
    
    curvatureAt: function( t ) {
      return ( this.anticlockwise ? -1 : 1 ) / this.radius;
    },
    
    positionAtAngle: function( angle ) {
      return this.center.plus( Vector2.createPolar( this.radius, angle ) );
    },
    
    tangentAtAngle: function( angle ) {
      var normal = Vector2.createPolar( 1, angle );
      
      return this.anticlockwise ? normal.perpendicular() : normal.perpendicular().negated();
    },
    
    // TODO: refactor? shared with Segment.EllipticalArc
    containsAngle: function( angle ) {
      // transform the angle into the appropriate coordinate form
      // TODO: check anticlockwise version!
      var normalizedAngle = this.anticlockwise ? angle - this.endAngle : angle - this.startAngle;
      
      // get the angle between 0 and 2pi
      var positiveMinAngle = normalizedAngle % ( Math.PI * 2 );
      // check this because modular arithmetic with negative numbers reveal a negative number
      if ( positiveMinAngle < 0 ) {
        positiveMinAngle += Math.PI * 2;
      }
      
      return positiveMinAngle <= this.angleDifference;
    },
    
    toPieces: function() {
      return [ new Piece.Arc( this.center, this.radius, this.startAngle, this.endAngle, this.anticlockwise ) ];
    },
    
    getSVGPathFragment: function() {
      // see http://www.w3.org/TR/SVG/paths.html#PathDataEllipticalArcCommands for more info
      // rx ry x-axis-rotation large-arc-flag sweep-flag x y
      
      var epsilon = 0.01; // allow some leeway to render things as 'almost circles'
      var sweepFlag = this.anticlockwise ? '0' : '1';
      var largeArcFlag;
      if ( this.angleDifference < Math.PI * 2 - epsilon ) {
        largeArcFlag = this.angleDifference < Math.PI ? '0' : '1';
        return 'A ' + this.radius + ' ' + this.radius + ' 0 ' + largeArcFlag + ' ' + sweepFlag + ' ' + this.end.x + ' ' + this.end.y;
      } else {
        // circle (or almost-circle) case needs to be handled differently
        // since SVG will not be able to draw (or know how to draw) the correct circle if we just have a start and end, we need to split it into two circular arcs
        
        // get the angle that is between and opposite of both of the points
        var splitOppositeAngle = ( this.startAngle + this.endAngle ) / 2; // this _should_ work for the modular case?
        var splitPoint = this.center.plus( Vector2.createPolar( this.radius, splitOppositeAngle ) );
        
        largeArcFlag = '0'; // since we split it in 2, it's always the small arc
        
        var firstArc = 'A ' + this.radius + ' ' + this.radius + ' 0 ' + largeArcFlag + ' ' + sweepFlag + ' ' + splitPoint.x + ' ' + splitPoint.y;
        var secondArc = 'A ' + this.radius + ' ' + this.radius + ' 0 ' + largeArcFlag + ' ' + sweepFlag + ' ' + this.end.x + ' ' + this.end.y;
        
        return firstArc + ' ' + secondArc;
      }
    },
    
    strokeLeft: function( lineWidth ) {
      return [ new Piece.Arc( this.center, this.radius + ( this.anticlockwise ? 1 : -1 ) * lineWidth / 2, this.startAngle, this.endAngle, this.anticlockwise ) ];
    },
    
    strokeRight: function( lineWidth ) {
      return [ new Piece.Arc( this.center, this.radius + ( this.anticlockwise ? -1 : 1 ) * lineWidth / 2, this.endAngle, this.startAngle, !this.anticlockwise ) ];
    },
    
    intersectsBounds: function( bounds ) {
      throw new Error( 'Segment.intersectsBounds unimplemented!' );
    },
    
    intersection: function( ray ) {
      var result = []; // hits in order
      
      // left here, if in the future we want to better-handle boundary points
      var epsilon = 0;
      
      // Run a general circle-intersection routine, then we can test the angles later.
      // Solves for the two solutions t such that ray.pos + ray.dir * t is on the circle.
      // Then we check whether the angle at each possible hit point is in our arc.
      var centerToRay = ray.pos.minus( this.center );
      var tmp = ray.dir.dot( centerToRay );
      var centerToRayDistSq = centerToRay.magnitudeSquared();
      var discriminant = 4 * tmp * tmp - 4 * ( centerToRayDistSq - this.radius * this.radius );
      if ( discriminant < epsilon ) {
        // ray misses circle entirely
        return result;
      }
      var base = ray.dir.dot( this.center ) - ray.dir.dot( ray.pos );
      var sqt = Math.sqrt( discriminant ) / 2;
      var ta = base - sqt;
      var tb = base + sqt;
      
      if ( tb < epsilon ) {
        // circle is behind ray
        return result;
      }
      
      var pointB = ray.pointAtDistance( tb );
      var normalB = pointB.minus( this.center ).normalized();
      
      if ( ta < epsilon ) {
        // we are inside the circle, so only one intersection is possible
        if ( this.containsAngle( normalB.angle() ) ) {
          result.push( {
            distance: tb,
            point: pointB,
            normal: normalB.negated(), // normal is towards the ray
            wind: this.anticlockwise ? -1 : 1 // since we are inside, wind this way
          } );
        }
      }
      else {
        // two possible hits (outside circle)
        var pointA = ray.pointAtDistance( ta );
        var normalA = pointA.minus( this.center ).normalized();
        
        if ( this.containsAngle( normalA.angle() ) ) {
          result.push( {
            distance: ta,
            point: pointA,
            normal: normalA,
            wind: this.anticlockwise ? 1 : -1 // hit from outside
          } );
        }
        if ( this.containsAngle( normalB.angle() ) ) {
          result.push( {
            distance: tb,
            point: pointB,
            normal: normalB.negated(),
            wind: this.anticlockwise ? -1 : 1 // this is the far hit, which winds the opposite way
          } );
        }
      }
      
      return result;
    },
    
    // returns the resultant winding number of this ray intersecting this segment.
    windingIntersection: function( ray ) {
      var wind = 0;
      var hits = this.intersection( ray );
      _.each( hits, function( hit ) {
        wind += hit.wind;
      } );
      return wind;
    }
  };
  
  return Segment.Arc;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Draws an arc.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/pieces/Arc',['require','ASSERT/assert','KITE/kite','DOT/Vector2','KITE/pieces/Piece','KITE/pieces/EllipticalArc','KITE/segments/Line','KITE/segments/Arc','KITE/util/Subpath'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );
  
  var kite = require( 'KITE/kite' );
  
  var Vector2 = require( 'DOT/Vector2' );
  
  var Piece = require( 'KITE/pieces/Piece' );
  require( 'KITE/pieces/EllipticalArc' );
  require( 'KITE/segments/Line' );
  require( 'KITE/segments/Arc' );
  require( 'KITE/util/Subpath' );
  
  Piece.Arc = function( center, radius, startAngle, endAngle, anticlockwise ) {
    if ( radius < 0 ) {
      // support this case since we might actually need to handle it inside of strokes?
      radius = -radius;
      startAngle += Math.PI;
      endAngle += Math.PI;
    }
    this.center = center;
    this.radius = radius;
    this.startAngle = startAngle;
    this.endAngle = endAngle;
    this.anticlockwise = anticlockwise;
  };
  Piece.Arc.prototype = {
    constructor: Piece.Arc,
    
    writeToContext: function( context ) {
      context.arc( this.center.x, this.center.y, this.radius, this.startAngle, this.endAngle, this.anticlockwise );
    },
    
    // TODO: test various transform types, especially rotations, scaling, shears, etc.
    transformed: function( matrix ) {
      // so we can handle reflections in the transform, we do the general case handling for start/end angles
      var startAngle = matrix.timesVector2( Vector2.createPolar( 1, this.startAngle ) ).minus( matrix.timesVector2( Vector2.ZERO ) ).angle();
      var endAngle = matrix.timesVector2( Vector2.createPolar( 1, this.endAngle ) ).minus( matrix.timesVector2( Vector2.ZERO ) ).angle();
      
      // reverse the 'clockwiseness' if our transform includes a reflection
      var anticlockwise = matrix.getDeterminant() >= 0 ? this.anticlockwise : !this.anticlockwise;

      var scaleVector = matrix.getScaleVector();
      if ( scaleVector.x !== scaleVector.y ) {
        var radiusX = scaleVector.x * this.radius;
        var radiusY = scaleVector.y * this.radius;
        return [new Piece.EllipticalArc( matrix.timesVector2( this.center ), radiusX, radiusY, 0, startAngle, endAngle, anticlockwise )];
      } else {
        var radius = scaleVector.x * this.radius;
        return [new Piece.Arc( matrix.timesVector2( this.center ), radius, startAngle, endAngle, anticlockwise )];
      }
    },
    
    applyPiece: function( shape ) {
      // see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-arc
      
      var arc = new kite.Segment.Arc( this.center, this.radius, this.startAngle, this.endAngle, this.anticlockwise );
      
      // we are assuming that the normal conditions were already met (or exceptioned out) so that these actually work with canvas
      var startPoint = arc.start;
      var endPoint = arc.end;
      
      // if there is already a point on the subpath, and it is different than our starting point, draw a line between them
      if ( shape.hasSubpaths() && shape.getLastSubpath().getLength() > 0 && !startPoint.equals( shape.getLastSubpath().getLastPoint(), 0 ) ) {
        shape.getLastSubpath().addSegment( new kite.Segment.Line( shape.getLastSubpath().getLastPoint(), startPoint ) );
      }
      
      if ( !shape.hasSubpaths() ) {
        shape.addSubpath( new kite.Subpath() );
      }
      
      shape.getLastSubpath().addSegment( arc );
      
      // technically the Canvas spec says to add the start point, so we do this even though it is probably completely unnecessary (there is no conditional)
      shape.getLastSubpath().addPoint( startPoint );
      shape.getLastSubpath().addPoint( endPoint );
      
      // and update the bounds
      if ( !arc.invalid ) {
        shape.bounds = shape.bounds.union( arc.bounds );
      }
    },
    
    toString: function() {
      return 'arc( ' + this.center.x + ', ' + this.center.y + ', ' + this.radius + ', ' + this.startAngle + ', ' + this.endAngle + ', ' + this.anticlockwise + ' )';
    }
  };
  
  return Piece.Arc;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Closes a subpath
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/pieces/Close',['require','ASSERT/assert','KITE/kite','KITE/pieces/Piece','KITE/util/Subpath'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );
  
  var kite = require( 'KITE/kite' );
  
  var Piece = require( 'KITE/pieces/Piece' );
  require( 'KITE/util/Subpath' );
  
  Piece.Close = function() {};
  Piece.Close.prototype = {
    constructor: Piece.Close,
    
    writeToContext: function( context ) {
      context.closePath();
    },
    
    transformed: function( matrix ) {
      return [this];
    },
    
    applyPiece: function( shape ) {
      if ( shape.hasSubpaths() ) {
        var previousPath = shape.getLastSubpath();
        var nextPath = new kite.Subpath();
        
        previousPath.close();
        shape.addSubpath( nextPath );
        nextPath.addPoint( previousPath.getFirstPoint() );
      }
    },
    
    toString: function() {
      return 'close()';
    }
  };
  
  return Piece.Close;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Quadratic Bezier segment
 *
 * Good reference: http://cagd.cs.byu.edu/~557/text/ch2.pdf
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/segments/Quadratic',['require','ASSERT/assert','KITE/kite','DOT/Bounds2','DOT/Matrix3','DOT/Util','KITE/segments/Segment','KITE/pieces/Piece'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );

  var kite = require( 'KITE/kite' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var solveQuadraticRootsReal = require( 'DOT/Util' ).solveQuadraticRootsReal;

  var Segment = require( 'KITE/segments/Segment' );
  var Piece = require( 'KITE/pieces/Piece' );

  Segment.Quadratic = function( start, control, end, skipComputations ) {
    this.start = start;
    this.control = control;
    this.end = end;
    
    if ( start.equals( end, 0 ) && start.equals( control, 0 ) ) {
      this.invalid = true;
      return;
    }
    
    var t;
    
    // allows us to skip unnecessary computation in the subdivision steps
    if ( skipComputations ) {
      return;
    }
    
    var controlIsStart = start.equals( control );
    var controlIsEnd = end.equals( control );
    // ensure the points are distinct
    assert && assert( !controlIsStart || !controlIsEnd );
    
    // allow either the start or end point to be the same as the control point (necessary if you do a quadraticCurveTo on an empty path)
    // tangents go through the control point, which simplifies things
    this.startTangent = controlIsStart ? end.minus( start ).normalized() : control.minus( start ).normalized();
    this.endTangent = controlIsEnd ? end.minus( start ).normalized() : end.minus( control ).normalized();
    
    // calculate our temporary guaranteed lower bounds based on the end points
    this.bounds = new Bounds2( Math.min( start.x, end.x ), Math.min( start.y, end.y ), Math.max( start.x, end.x ), Math.max( start.y, end.y ) );
    
    // compute x and y where the derivative is 0, so we can include this in the bounds
    var divisorX = 2 * ( end.x - 2 * control.x + start.x );
    if ( divisorX !== 0 ) {
      t = -2 * ( control.x - start.x ) / divisorX;
      
      if ( t > 0 && t < 1 ) {
        this.bounds = this.bounds.withPoint( this.positionAt( t ) );
      }
    }
    var divisorY = 2 * ( end.y - 2 * control.y + start.y );
    if ( divisorY !== 0 ) {
      t = -2 * ( control.y - start.y ) / divisorY;
      
      if ( t > 0 && t < 1 ) {
        this.bounds = this.bounds.withPoint( this.positionAt( t ) );
      }
    }
  };
  Segment.Quadratic.prototype = {
    constructor: Segment.Quadratic,
    
    degree: 2,
    
    // can be described from t=[0,1] as: (1-t)^2 start + 2(1-t)t control + t^2 end
    positionAt: function( t ) {
      var mt = 1 - t;
      return this.start.times( mt * mt ).plus( this.control.times( 2 * mt * t ) ).plus( this.end.times( t * t ) );
    },
    
    // derivative: 2(1-t)( control - start ) + 2t( end - control )
    tangentAt: function( t ) {
      return this.control.minus( this.start ).times( 2 * ( 1 - t ) ).plus( this.end.minus( this.control ).times( 2 * t ) );
    },
    
    curvatureAt: function( t ) {
      // see http://cagd.cs.byu.edu/~557/text/ch2.pdf p31
      // TODO: remove code duplication with Cubic
      var epsilon = 0.0000001;
      if ( Math.abs( t - 0.5 ) > 0.5 - epsilon ) {
        var isZero = t < 0.5;
        var p0 = isZero ? this.start : this.end;
        var p1 = this.control;
        var p2 = isZero ? this.end : this.start;
        var d10 = p1.minus( p0 );
        var a = d10.magnitude();
        var h = ( isZero ? -1 : 1 ) * d10.perpendicular().normalized().dot( p2.minus( p1 ) );
        return ( h * ( this.degree - 1 ) ) / ( this.degree * a * a );
      } else {
        return this.subdivided( t, true )[0].curvatureAt( 1 );
      }
    },
    
    // see http://www.visgraf.impa.br/sibgrapi96/trabs/pdf/a14.pdf
    // and http://math.stackexchange.com/questions/12186/arc-length-of-bezier-curves for curvature / arc length
    
    offsetTo: function( r, reverse ) {
      // TODO: implement more accurate method at http://www.antigrain.com/research/adaptive_bezier/index.html
      // TODO: or more recently (and relevantly): http://www.cis.usouthal.edu/~hain/general/Publications/Bezier/BezierFlattening.pdf
      var curves = [this];
      
      // subdivide this curve
      var depth = 5; // generates 2^depth curves
      for ( var i = 0; i < depth; i++ ) {
        curves = _.flatten( _.map( curves, function( curve ) {
          return curve.subdivided( 0.5, true );
        } ));
      }
      
      var offsetCurves = _.map( curves, function( curve ) { return curve.approximateOffset( r ); } );
      
      if ( reverse ) {
        offsetCurves.reverse();
        offsetCurves = _.map( offsetCurves, function( curve ) { return curve.reversed( true ); } );
      }
      
      var result = _.map( offsetCurves, function( curve ) {
        return new Piece.QuadraticCurveTo( curve.control, curve.end );
      } );
      
      return result;
    },
    
    subdivided: function( t, skipComputations ) {
      // de Casteljau method
      var leftMid = this.start.blend( this.control, t );
      var rightMid = this.control.blend( this.end, t );
      var mid = leftMid.blend( rightMid, t );
      return [
        new Segment.Quadratic( this.start, leftMid, mid, skipComputations ),
        new Segment.Quadratic( mid, rightMid, this.end, skipComputations )
      ];
    },
    
    reversed: function( skipComputations ) {
      return new Segment.Quadratic( this.end, this.control, this.start );
    },
    
    approximateOffset: function( r ) {
      return new Segment.Quadratic(
        this.start.plus( ( this.start.equals( this.control ) ? this.end.minus( this.start ) : this.control.minus( this.start ) ).perpendicular().normalized().times( r ) ),
        this.control.plus( this.end.minus( this.start ).perpendicular().normalized().times( r ) ),
        this.end.plus( ( this.end.equals( this.control ) ? this.end.minus( this.start ) : this.end.minus( this.control ) ).perpendicular().normalized().times( r ) )
      );
    },
    
    toPieces: function() {
      return [ new Piece.QuadraticCurveTo( this.control, this.end ) ];
    },
    
    getSVGPathFragment: function() {
      return 'Q ' + this.control.x + ' ' + this.control.y + ' ' + this.end.x + ' ' + this.end.y;
    },
    
    strokeLeft: function( lineWidth ) {
      return this.offsetTo( -lineWidth / 2, false );
    },
    
    strokeRight: function( lineWidth ) {
      return this.offsetTo( lineWidth / 2, true );
    },
    
    intersectsBounds: function( bounds ) {
      throw new Error( 'Segment.Quadratic.intersectsBounds unimplemented' ); // TODO: implement
    },
    
    // returns the resultant winding number of this ray intersecting this segment.
    intersection: function( ray ) {
      var self = this;
      var result = [];
      
      // find the rotation that will put our ray in the direction of the x-axis so we can only solve for y=0 for intersections
      var inverseMatrix = Matrix3.rotation2( -ray.dir.angle() ).timesMatrix( Matrix3.translation( -ray.pos.x, -ray.pos.y ) );
      
      var p0 = inverseMatrix.timesVector2( this.start );
      var p1 = inverseMatrix.timesVector2( this.control );
      var p2 = inverseMatrix.timesVector2( this.end );
      
      //(1-t)^2 start + 2(1-t)t control + t^2 end
      var a = p0.y - 2 * p1.y + p2.y;
      var b = -2 * p0.y + 2 * p1.y;
      var c = p0.y;
      
      var ts = solveQuadraticRootsReal( a, b, c );
      
      _.each( ts, function( t ) {
        if ( t >= 0 && t <= 1 ) {
          var hitPoint = self.positionAt( t );
          var unitTangent = self.tangentAt( t ).normalized();
          var perp = unitTangent.perpendicular();
          var toHit = hitPoint.minus( ray.pos );
          
          // make sure it's not behind the ray
          if ( toHit.dot( ray.dir ) > 0 ) {
            result.push( {
              distance: toHit.magnitude(),
              point: hitPoint,
              normal: perp.dot( ray.dir ) > 0 ? perp.negated() : perp,
              wind: ray.dir.perpendicular().dot( unitTangent ) < 0 ? 1 : -1
            } );
          }
        }
      } );
      return result;
    },
    
    windingIntersection: function( ray ) {
      var wind = 0;
      var hits = this.intersection( ray );
      _.each( hits, function( hit ) {
        wind += hit.wind;
      } );
      return wind;
    }
  };
  
  return Segment.Quadratic;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Cubic Bezier segment.
 *
 * See http://www.cis.usouthal.edu/~hain/general/Publications/Bezier/BezierFlattening.pdf for info
 *
 * Good reference: http://cagd.cs.byu.edu/~557/text/ch2.pdf
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/segments/Cubic',['require','ASSERT/assert','KITE/kite','DOT/Bounds2','DOT/Vector2','DOT/Matrix3','DOT/Util','DOT/Util','KITE/segments/Segment','KITE/pieces/Piece','KITE/segments/Quadratic'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );

  var kite = require( 'KITE/kite' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Vector2 = require( 'DOT/Vector2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var solveQuadraticRootsReal = require( 'DOT/Util' ).solveQuadraticRootsReal;
  var solveCubicRootsReal = require( 'DOT/Util' ).solveCubicRootsReal;
  
  var Segment = require( 'KITE/segments/Segment' );
  var Piece = require( 'KITE/pieces/Piece' );
  require( 'KITE/segments/Quadratic' );

  Segment.Cubic = function( start, control1, control2, end, skipComputations ) {
    this.start = start;
    this.control1 = control1;
    this.control2 = control2;
    this.end = end;
    
    // allows us to skip unnecessary computation in the subdivision steps
    if ( skipComputations ) {
      return;
    }
    
    this.startTangent = this.tangentAt( 0 ).normalized();
    this.endTangent = this.tangentAt( 1 ).normalized();
    
    if ( start.equals( end, 0 ) && start.equals( control1, 0 ) && start.equals( control2, 0 ) ) {
      this.invalid = true;
      return;
    }
    
    // from http://www.cis.usouthal.edu/~hain/general/Publications/Bezier/BezierFlattening.pdf
    this.r = control1.minus( start ).normalized();
    this.s = this.r.perpendicular();
    
    var a = start.times( -1 ).plus( control1.times( 3 ) ).plus( control2.times( -3 ) ).plus( end );
    var b = start.times( 3 ).plus( control1.times( -6 ) ).plus( control2.times( 3 ) );
    var c = start.times( -3 ).plus( control1.times( 3 ) );
    var d = start;
    
    var aPerp = a.perpendicular();
    var bPerp = b.perpendicular();
    var aPerpDotB = aPerp.dot( b );
    
    this.tCusp = -0.5 * ( aPerp.dot( c ) / aPerpDotB );
    this.tDeterminant = this.tCusp * this.tCusp - ( 1 / 3 ) * ( bPerp.dot( c ) / aPerpDotB );
    if ( this.tDeterminant >= 0 ) {
      var sqrtDet = Math.sqrt( this.tDeterminant );
      this.tInflection1 = this.tCusp - sqrtDet;
      this.tInflection2 = this.tCusp + sqrtDet;
    }
    
    if ( this.hasCusp() ) {
      // if there is a cusp, we'll split at the cusp into two quadratic bezier curves.
      // see http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.94.8088&rep=rep1&type=pdf (Singularities of rational Bezier curves - J Monterde, 2001)
      var subdividedAtCusp = this.subdivided( this.tCusp, true );
      this.startQuadratic = new Segment.Quadratic( subdividedAtCusp[0].start, subdividedAtCusp[0].control1, subdividedAtCusp[0].end, false );
      this.endQuadratic = new Segment.Quadratic( subdividedAtCusp[1].start, subdividedAtCusp[1].control2, subdividedAtCusp[1].end, false );
    }
    
    this.bounds = Bounds2.NOTHING;
    this.bounds = this.bounds.withPoint( this.start );
    this.bounds = this.bounds.withPoint( this.end );
    
    /*---------------------------------------------------------------------------*
    * Bounds
    *----------------------------------------------------------------------------*/
    
    // finds what t values the cubic extrema are at (if any).
    function extremaT( v0, v1, v2, v3 ) {
      // coefficients of derivative
      var a = -3 * v0 + 9 * v1 -9 * v2 + 3 * v3;
      var b =  6 * v0 - 12 * v1 + 6 * v2;
      var c = -3 * v0 + 3 * v1;
      
      return solveQuadraticRootsReal( a, b, c );
    }
    
    var cubic = this;
    _.each( extremaT( this.start.x, this.control1.x, this.control2.x, this.end.x ), function( t ) {
      if ( t >= 0 && t <= 1 ) {
        cubic.bounds = cubic.bounds.withPoint( cubic.positionAt( t ) );
      }
    } );
    _.each( extremaT( this.start.y, this.control1.y, this.control2.y, this.end.y ), function( t ) {
      if ( t >= 0 && t <= 1 ) {
        cubic.bounds = cubic.bounds.withPoint( cubic.positionAt( t ) );
      }
    } );
    
    if ( this.hasCusp() ) {
      this.bounds = this.bounds.withPoint( this.positionAt( this.tCusp ) );
    }
  };
  Segment.Cubic.prototype = {
    constructor: Segment.Cubic,
    
    degree: 3,
    
    hasCusp: function() {
      var epsilon = 0.000001; // TODO: make this available to change?
      return this.tangentAt( this.tCusp ).magnitude() < epsilon && this.tCusp >= 0 && this.tCusp <= 1;
    },
    
    // position: (1 - t)^3*start + 3*(1 - t)^2*t*control1 + 3*(1 - t) t^2*control2 + t^3*end
    positionAt: function( t ) {
      var mt = 1 - t;
      return this.start.times( mt * mt * mt ).plus( this.control1.times( 3 * mt * mt * t ) ).plus( this.control2.times( 3 * mt * t * t ) ).plus( this.end.times( t * t * t ) );
    },
    
    // derivative: -3 p0 (1 - t)^2 + 3 p1 (1 - t)^2 - 6 p1 (1 - t) t + 6 p2 (1 - t) t - 3 p2 t^2 + 3 p3 t^2
    tangentAt: function( t ) {
      var mt = 1 - t;
      return this.start.times( -3 * mt * mt ).plus( this.control1.times( 3 * mt * mt - 6 * mt * t ) ).plus( this.control2.times( 6 * mt * t - 3 * t * t ) ).plus( this.end.times( 3 * t * t ) );
    },
    
    curvatureAt: function( t ) {
      // see http://cagd.cs.byu.edu/~557/text/ch2.pdf p31
      // TODO: remove code duplication with Quadratic
      var epsilon = 0.0000001;
      if ( Math.abs( t - 0.5 ) > 0.5 - epsilon ) {
        var isZero = t < 0.5;
        var p0 = isZero ? this.start : this.end;
        var p1 = isZero ? this.control1 : this.control2;
        var p2 = isZero ? this.control2 : this.control1;
        var d10 = p1.minus( p0 );
        var a = d10.magnitude();
        var h = ( isZero ? -1 : 1 ) * d10.perpendicular().normalized().dot( p2.minus( p1 ) );
        return ( h * ( this.degree - 1 ) ) / ( this.degree * a * a );
      } else {
        return this.subdivided( t, true )[0].curvatureAt( 1 );
      }
    },
    
    toRS: function( point ) {
      var firstVector = point.minus( this.start );
      return new Vector2( firstVector.dot( this.r ), firstVector.dot( this.s ) );
    },
    
    subdivided: function( t, skipComputations ) {
      // de Casteljau method
      // TODO: add a 'bisect' or 'between' method for vectors?
      var left = this.start.blend( this.control1, t );
      var right = this.control2.blend( this.end, t );
      var middle = this.control1.blend( this.control2, t );
      var leftMid = left.blend( middle, t );
      var rightMid = middle.blend( right, t );
      var mid = leftMid.blend( rightMid, t );
      return [
        new Segment.Cubic( this.start, left, leftMid, mid, skipComputations ),
        new Segment.Cubic( mid, rightMid, right, this.end, skipComputations )
      ];
    },
    
    offsetTo: function( r, reverse ) {
      // TODO: implement more accurate method at http://www.antigrain.com/research/adaptive_bezier/index.html
      // TODO: or more recently (and relevantly): http://www.cis.usouthal.edu/~hain/general/Publications/Bezier/BezierFlattening.pdf
      
      // how many segments to create (possibly make this more adaptive?)
      var quantity = 32;
      
      var result = [];
      for ( var i = 1; i < quantity; i++ ) {
        var t = i / ( quantity - 1 );
        if ( reverse ) {
          t = 1 - t;
        }
        
        var point = this.positionAt( t ).plus( this.tangentAt( t ).perpendicular().normalized().times( r ) );
        result.push( new Piece.LineTo( point ) );
      }
      
      return result;
    },
    
    toPieces: function() {
      return [ new Piece.CubicCurveTo( this.control1, this.control2, this.end ) ];
    },
    
    getSVGPathFragment: function() {
      return 'C ' + this.control1.x + ' ' + this.control1.y + ' ' + this.control2.x + ' ' + this.control2.y + ' ' + this.end.x + ' ' + this.end.y;
    },
    
    strokeLeft: function( lineWidth ) {
      return this.offsetTo( -lineWidth / 2, false );
    },
    
    strokeRight: function( lineWidth ) {
      return this.offsetTo( lineWidth / 2, true );
    },
    
    intersectsBounds: function( bounds ) {
      throw new Error( 'Segment.Cubic.intersectsBounds unimplemented' ); // TODO: implement
    },
    
    // returns the resultant winding number of this ray intersecting this segment.
    intersection: function( ray ) {
      var self = this;
      var result = [];
      
      // find the rotation that will put our ray in the direction of the x-axis so we can only solve for y=0 for intersections
      var inverseMatrix = Matrix3.rotation2( -ray.dir.angle() ).timesMatrix( Matrix3.translation( -ray.pos.x, -ray.pos.y ) );
      
      var p0 = inverseMatrix.timesVector2( this.start );
      var p1 = inverseMatrix.timesVector2( this.control1 );
      var p2 = inverseMatrix.timesVector2( this.control2 );
      var p3 = inverseMatrix.timesVector2( this.end );
      
      // polynomial form of cubic: start + (3 control1 - 3 start) t + (-6 control1 + 3 control2 + 3 start) t^2 + (3 control1 - 3 control2 + end - start) t^3
      var a = -p0.y + 3 * p1.y - 3 * p2.y + p3.y;
      var b = 3 * p0.y - 6 * p1.y + 3 * p2.y;
      var c = -3 * p0.y + 3 * p1.y;
      var d = p0.y;
      
      var ts = solveCubicRootsReal( a, b, c, d );
      
      _.each( ts, function( t ) {
        if ( t >= 0 && t <= 1 ) {
          var hitPoint = self.positionAt( t );
          var unitTangent = self.tangentAt( t ).normalized();
          var perp = unitTangent.perpendicular();
          var toHit = hitPoint.minus( ray.pos );
          
          // make sure it's not behind the ray
          if ( toHit.dot( ray.dir ) > 0 ) {
            result.push( {
              distance: toHit.magnitude(),
              point: hitPoint,
              normal: perp.dot( ray.dir ) > 0 ? perp.negated() : perp,
              wind: ray.dir.perpendicular().dot( unitTangent ) < 0 ? 1 : -1
            } );
          }
        }
      } );
      return result;
    },
    
    windingIntersection: function( ray ) {
      var wind = 0;
      var hits = this.intersection( ray );
      _.each( hits, function( hit ) {
        wind += hit.wind;
      } );
      return wind;
    }
    
    // returns the resultant winding number of this ray intersecting this segment.
    // windingIntersection: function( ray ) {
    //   // find the rotation that will put our ray in the direction of the x-axis so we can only solve for y=0 for intersections
    //   var inverseMatrix = Matrix3.rotation2( -ray.dir.angle() );
    //   assert && assert( inverseMatrix.timesVector2( ray.dir ).x > 0.99 ); // verify that we transform the unit vector to the x-unit
      
    //   var y0 = inverseMatrix.timesVector2( this.start ).y;
    //   var y1 = inverseMatrix.timesVector2( this.control1 ).y;
    //   var y2 = inverseMatrix.timesVector2( this.control2 ).y;
    //   var y3 = inverseMatrix.timesVector2( this.end ).y;
      
    //   // polynomial form of cubic: start + (3 control1 - 3 start) t + (-6 control1 + 3 control2 + 3 start) t^2 + (3 control1 - 3 control2 + end - start) t^3
    //   var a = -y0 + 3 * y1 - 3 * y2 + y3;
    //   var b = 3 * y0 - 6 * y1 + 3 * y2;
    //   var c = -3 * y0 + 3 * y1;
    //   var d = y0;
      
    //   // solve cubic roots
    //   var ts = solveCubicRootsReal( a, b, c, d );
      
    //   var result = 0;
      
    //   // for each hit
    //   _.each( ts, function( t ) {
    //     if ( t >= 0 && t <= 1 ) {
    //       result += ray.dir.perpendicular().dot( this.tangentAt( t ) ) < 0 ? 1 : -1;
    //     }
    //   } );
      
    //   return result;
    // }
  };
  
  return Segment.Cubic;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Draws a cubic bezier curve
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/pieces/CubicCurveTo',['require','ASSERT/assert','KITE/kite','KITE/pieces/Piece','KITE/segments/Cubic'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );
  
  var kite = require( 'KITE/kite' );
  
  var Piece = require( 'KITE/pieces/Piece' );
  require( 'KITE/segments/Cubic' );
  
  Piece.CubicCurveTo = function( control1, control2, point ) {
    this.control1 = control1;
    this.control2 = control2;
    this.point = point;
  };
  Piece.CubicCurveTo.prototype = {
    constructor: Piece.CubicCurveTo,
    
    writeToContext: function( context ) {
      context.bezierCurveTo( this.control1.x, this.control1.y, this.control2.x, this.control2.y, this.point.x, this.point.y );
    },
    
    transformed: function( matrix ) {
      return [new Piece.CubicCurveTo( matrix.timesVector2( this.control1 ), matrix.timesVector2( this.control2 ), matrix.timesVector2( this.point ) )];
    },
    
    applyPiece: function( shape ) {
      // see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-quadraticcurveto
      shape.ensure( this.controlPoint );
      var start = shape.getLastSubpath().getLastPoint();
      var cubic = new kite.Segment.Cubic( start, this.control1, this.control2, this.point );
      
      // if there is a cusp, we add the two (split) quadratic segments instead so that stroking treats the 'join' between them with the proper lineJoin
      if ( cubic.hasCusp() ) {
        shape.getLastSubpath().addSegment( cubic.startQuadratic );
        shape.getLastSubpath().addSegment( cubic.endQuadratic );
      } else {
        shape.getLastSubpath().addSegment( cubic );
      }
      shape.getLastSubpath().addPoint( this.point );
      if ( !cubic.invalid ) {
        shape.bounds = shape.bounds.union( cubic.bounds );
      }
    },
    
    toString: function() {
      return 'cubicCurveTo( ' + this.control1.x + ', ' + this.control1.y + ', ' + this.control2.x + ', ' + this.control2.y + ', ' + this.point.x + ', ' + this.point.y + ' )';
    }
  };
  
  return Piece.CubicCurveTo;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Creates a line from the previous point
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/pieces/LineTo',['require','ASSERT/assert','KITE/kite','KITE/pieces/Piece','KITE/segments/Line'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );
  
  var kite = require( 'KITE/kite' );
  
  var Piece = require( 'KITE/pieces/Piece' );
  require( 'KITE/segments/Line' );
  
  Piece.LineTo = function( point ) {
    this.point = point;
  };
  Piece.LineTo.prototype = {
    constructor: Piece.LineTo,
    
    writeToContext: function( context ) {
      context.lineTo( this.point.x, this.point.y );
    },
    
    transformed: function( matrix ) {
      return [new Piece.LineTo( matrix.timesVector2( this.point ) )];
    },
    
    applyPiece: function( shape ) {
      // see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-lineto
      if ( shape.hasSubpaths() ) {
        var start = shape.getLastSubpath().getLastPoint();
        var end = this.point;
        var line = new kite.Segment.Line( start, end );
        shape.getLastSubpath().addSegment( line );
        shape.getLastSubpath().addPoint( end );
        shape.bounds = shape.bounds.withPoint( start ).withPoint( end );
        assert && assert( !isNaN( shape.bounds.getX() ) );
      } else {
        shape.ensure( this.point );
      }
    },
    
    toString: function() {
      return 'lineTo( ' + this.point.x + ', ' + this.point.y + ' )';
    }
  };
  
  return Piece.LineTo;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Creates a new subpath starting at the specified point
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/pieces/MoveTo',['require','ASSERT/assert','KITE/kite','KITE/pieces/Piece','KITE/util/Subpath'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );
  
  var kite = require( 'KITE/kite' );
  
  var Piece = require( 'KITE/pieces/Piece' );
  require( 'KITE/util/Subpath' );
  
  Piece.MoveTo = function( point ) {
    this.point = point;
  };
  Piece.MoveTo.prototype = {
    constructor: Piece.MoveTo,
    
    writeToContext: function( context ) {
      context.moveTo( this.point.x, this.point.y );
    },
    
    transformed: function( matrix ) {
      return [new Piece.MoveTo( matrix.timesVector2( this.point ) )];
    },
    
    applyPiece: function( shape ) {
      var subpath = new kite.Subpath();
      subpath.addPoint( this.point );
      shape.addSubpath( subpath );
    },
    
    toString: function() {
      return 'moveTo( ' + this.point.x + ', ' + this.point.y + ' )';
    }
  };
  
  return Piece.MoveTo;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Draws a quadratic bezier curve
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/pieces/QuadraticCurveTo',['require','ASSERT/assert','KITE/kite','KITE/pieces/Piece','KITE/segments/Quadratic'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );
  
  var kite = require( 'KITE/kite' );
  
  var Piece = require( 'KITE/pieces/Piece' );
  require( 'KITE/segments/Quadratic' );
  
  Piece.QuadraticCurveTo = function( controlPoint, point ) {
    this.controlPoint = controlPoint;
    this.point = point;
  };
  Piece.QuadraticCurveTo.prototype = {
    constructor: Piece.QuadraticCurveTo,
    
    writeToContext: function( context ) {
      context.quadraticCurveTo( this.controlPoint.x, this.controlPoint.y, this.point.x, this.point.y );
    },
    
    transformed: function( matrix ) {
      return [new Piece.QuadraticCurveTo( matrix.timesVector2( this.controlPoint ), matrix.timesVector2( this.point ) )];
    },
    
    applyPiece: function( shape ) {
      // see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-quadraticcurveto
      shape.ensure( this.controlPoint );
      var start = shape.getLastSubpath().getLastPoint();
      var quadratic = new kite.Segment.Quadratic( start, this.controlPoint, this.point );
      shape.getLastSubpath().addSegment( quadratic );
      shape.getLastSubpath().addPoint( this.point );
      if ( !quadratic.invalid ) {
        shape.bounds = shape.bounds.union( quadratic.bounds );
      }
    },
    
    toString: function() {
      return 'quadraticCurveTo( ' + this.controlPoint.x + ', ' + this.controlPoint.y + ', ' + this.point.x + ', ' + this.point.y + ' )';
    }
  };
  
  return Piece.QuadraticCurveTo;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Draws a rectangle.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/pieces/Rect',['require','ASSERT/assert','KITE/kite','DOT/Vector2','KITE/pieces/Piece','KITE/pieces/MoveTo','KITE/pieces/Close','KITE/util/Subpath','KITE/segments/Line'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );
  
  var kite = require( 'KITE/kite' );
  
  var Vector2 = require( 'DOT/Vector2' );
  
  var Piece = require( 'KITE/pieces/Piece' );
  require( 'KITE/pieces/MoveTo' );
  require( 'KITE/pieces/Close' );
  require( 'KITE/util/Subpath' );
  require( 'KITE/segments/Line' );
  
  // for brevity
  function p( x,y ) { return new Vector2( x, y ); }
  
  Piece.Rect = function( x, y, width, height ) {
    assert && assert( x !== undefined && y !== undefined && width !== undefined && height !== undefined, 'Undefined argument for Piece.Rect' );
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
  };
  Piece.Rect.prototype = {
    constructor: Piece.Rect,
    
    writeToContext: function( context ) {
      context.rect( this.x, this.y, this.width, this.height );
    },
    
    transformed: function( matrix ) {
      var a = matrix.timesVector2( p( this.x, this.y ) );
      var b = matrix.timesVector2( p( this.x + this.width, this.y ) );
      var c = matrix.timesVector2( p( this.x + this.width, this.y + this.height ) );
      var d = matrix.timesVector2( p( this.x, this.y + this.height ) );
      return [new Piece.MoveTo( a ), new Piece.LineTo( b ), new Piece.LineTo( c ), new Piece.LineTo( d ), new Piece.Close(), new Piece.MoveTo( a )];
    },
    
    applyPiece: function( shape ) {
      var subpath = new kite.Subpath();
      shape.addSubpath( subpath );
      subpath.addPoint( p( this.x, this.y ) );
      subpath.addPoint( p( this.x + this.width, this.y ) );
      subpath.addPoint( p( this.x + this.width, this.y + this.height ) );
      subpath.addPoint( p( this.x, this.y + this.height ) );
      subpath.addSegment( new kite.Segment.Line( subpath.points[0], subpath.points[1] ) );
      subpath.addSegment( new kite.Segment.Line( subpath.points[1], subpath.points[2] ) );
      subpath.addSegment( new kite.Segment.Line( subpath.points[2], subpath.points[3] ) );
      subpath.close();
      shape.addSubpath( new kite.Subpath() );
      shape.getLastSubpath().addPoint( p( this.x, this.y ) );
      shape.bounds = shape.bounds.withCoordinates( this.x, this.y ).withCoordinates( this.x + this.width, this.y + this.height );
      assert && assert( !isNaN( shape.bounds.getX() ) );
    },
    
    toString: function() {
      return 'rect( ' + this.x + ', ' + this.y + ', ' + this.width + ', ' + this.height + ' )';
    }
  };
  
  return Piece.Rect;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Shape handling
 *
 * Shapes are internally made up of pieces (generally individual Canvas calls),
 * which for simplicity of stroking and hit testing are then broken up into
 * individual segments stored in subpaths. Familiarity with how Canvas handles
 * subpaths is helpful for understanding this code.
 *
 * Canvas spec: http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html
 * SVG spec: http://www.w3.org/TR/SVG/expanded-toc.html
 *           http://www.w3.org/TR/SVG/paths.html#PathData (for paths)
 * Notes for elliptical arcs: http://www.w3.org/TR/SVG/implnote.html#PathElementImplementationNotes
 * Notes for painting strokes: https://svgwg.org/svg2-draft/painting.html
 *
 * TODO: add nonzero / evenodd support when browsers support it
 * TODO: docs
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/Shape',['require','ASSERT/assert','ASSERT/assert','KITE/kite','DOT/Vector2','DOT/Bounds2','DOT/Ray2','DOT/Matrix3','DOT/Transform3','DOT/Util','DOT/Util','KITE/util/Subpath','KITE/pieces/Piece','KITE/util/LineStyles','KITE/pieces/Arc','KITE/pieces/Close','KITE/pieces/CubicCurveTo','KITE/pieces/EllipticalArc','KITE/pieces/LineTo','KITE/pieces/MoveTo','KITE/pieces/QuadraticCurveTo','KITE/pieces/Rect','KITE/segments/Line'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'kite' );
  var assertExtra = require( 'ASSERT/assert' )( 'kite.extra', true );
  
  var kite = require( 'KITE/kite' );
  
  var Vector2 = require( 'DOT/Vector2' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Ray2 = require( 'DOT/Ray2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Transform3 = require( 'DOT/Transform3' );
  var toDegrees = require( 'DOT/Util' ).toDegrees;
  var lineLineIntersection = require( 'DOT/Util' ).lineLineIntersection;
  
  var Subpath = require( 'KITE/util/Subpath' );
  var Piece = require( 'KITE/pieces/Piece' );
  require( 'KITE/util/LineStyles' );
  require( 'KITE/pieces/Arc' );
  require( 'KITE/pieces/Close' );
  require( 'KITE/pieces/CubicCurveTo' );
  require( 'KITE/pieces/EllipticalArc' );
  require( 'KITE/pieces/LineTo' );
  require( 'KITE/pieces/MoveTo' );
  require( 'KITE/pieces/QuadraticCurveTo' );
  require( 'KITE/pieces/Rect' );
  require( 'KITE/segments/Line' );
  
  // for brevity
  function p( x,y ) { return new Vector2( x, y ); }
  
  // a normalized vector for non-zero winding checks
  // var weirdDir = p( Math.PI, 22 / 7 );
  
  kite.Shape = function( pieces, optionalClose ) {
    // higher-level Canvas-esque drawing commands, individually considered to be immutable
    this.pieces = [];
    
    // lower-level piecewise mathematical description using segments, also individually immutable
    this.subpaths = [];
    
    // computed bounds for all pieces added so far
    this.bounds = Bounds2.NOTHING;
    
    // cached stroked shape (so hit testing can be done quickly on stroked shapes)
    this._strokedShape = null;
    this._strokedShapeComputed = false;
    this._strokedShapeStyles = null;
    
    var that = this;
    // initialize with pieces passed in
    if ( pieces !== undefined ) {
      _.each( pieces, function( piece ) {
        that.addPiece( piece );
      } );
    }
    if ( optionalClose ) {
      this.addPiece( new Piece.Close() );
    }
  };
  var Shape = kite.Shape;
  
  Shape.prototype = {
    constructor: Shape,
    
    moveTo: function( x, y ) {
      // moveTo( point )
      if ( y === undefined && typeof x === 'object' ) {
        // wrap it in a Vector2 if the class doesn't match
        var point = x instanceof Vector2 ? x : new Vector2( x.x, x.y );
        this.addPiece( new Piece.MoveTo( point ) );
      } else { // moveTo( x, y )
        this.addPiece( new Piece.MoveTo( p( x, y ) ) );
      }
      return this;
    },
    
    lineTo: function( x, y ) {
      // lineTo( point )
      if ( y === undefined && typeof x === 'object' ) {
        // wrap it in a Vector2 if the class doesn't match
        var point = x instanceof Vector2 ? x : new Vector2( x.x, x.y );
        this.addPiece( new Piece.LineTo( point ) );
      } else { // lineTo( x, y )
        this.addPiece( new Piece.LineTo( p( x, y ) ) );
      }
      return this;
    },
    
    quadraticCurveTo: function( cpx, cpy, x, y ) {
      // quadraticCurveTo( control, point )
      if ( x === undefined && typeof cpx === 'object' ) {
        // wrap it in a Vector2 if the class doesn't match
        var controlPoint = cpx instanceof Vector2 ? cpx : new Vector2( cpx.x, cpx.y );
        var point = cpy instanceof Vector2 ? cpy : new Vector2( cpy.x, cpy.y );
        this.addPiece( new Piece.QuadraticCurveTo( controlPoint, point ) );
      } else { // quadraticCurveTo( cpx, cpy, x, y )
        this.addPiece( new Piece.QuadraticCurveTo( p( cpx, cpy ), p( x, y ) ) );
      }
      return this;
    },
    
    cubicCurveTo: function( cp1x, cp1y, cp2x, cp2y, x, y ) {
      // cubicCurveTo( cp1, cp2, end )
      if ( cp2y === undefined && typeof cp1x === 'object' ) {
        // wrap it in a Vector2 if the class doesn't match
        var control1 = cp1x instanceof Vector2 ? cp1x : new Vector2( cp1x.x, cp1x.y );
        var control2 = cp1y instanceof Vector2 ? cp1y : new Vector2( cp1y.x, cp1y.y );
        var end = cp2x instanceof Vector2 ? cp2x : new Vector2( cp2x.x, cp2x.y );
        this.addPiece( new Piece.CubicCurveTo( control1, control2, end ) );
      } else { // cubicCurveTo( cp1x, cp1y, cp2x, cp2y, x, y )
        this.addPiece( new Piece.CubicCurveTo( p( cp1x, cp1y ), p( cp2x, cp2y ), p( x, y ) ) );
      }
      return this;
    },
    
    /*
     * Draws a circle using the arc() call with the following parameters:
     * circle( center, radius ) // center is a Vector2
     * circle( centerX, centerY, radius )
     */
    circle: function( centerX, centerY, radius ) {
      if ( typeof centerX === 'object' ) {
        // circle( center, radius )
        var center = centerX;
        radius = centerY;
        return this.arc( center, radius, 0, Math.PI * 2, false );
      } else {
        // circle( centerX, centerY, radius )
        return this.arc( p( centerX, centerY ), radius, 0, Math.PI * 2, false );
      }
    },
    
    /*
     * Draws an ellipse using the ellipticalArc() call with the following parameters:
     * ellipse( center, radiusX, radiusY, rotation ) // center is a Vector2
     * ellipse( centerX, centerY, radiusX, radiusY, rotation )
     */
    ellipse: function( centerX, centerY, radiusX, radiusY, rotation ) {
      // TODO: Ellipse/EllipticalArc has a mess of parameters. Consider parameter object, or double-check parameter handling
      if ( typeof centerX === 'object' ) {
        // ellipse( center, radiusX, radiusY, rotation )
        var center = centerX;
        rotation = radiusY;
        radiusY = radiusX;
        radiusX = centerY;
        return this.ellipticalArc( center, radiusX, radiusY, rotation || 0, 0, Math.PI * 2, false );
      } else {
        // ellipse( centerX, centerY, radiusX, radiusY, rotation )
        return this.ellipticalArc( p( centerX, centerY ), radiusX, radiusY, rotation || 0, 0, Math.PI * 2, false );
      }
    },
    
    /*
     * Draws an arc using the Canvas 2D semantics, with the following parameters:
     * arc( center, radius, startAngle, endAngle, anticlockwise )
     * arc( centerX, centerY, radius, startAngle, endAngle, anticlockwise )
     */
    arc: function( centerX, centerY, radius, startAngle, endAngle, anticlockwise ) {
      if ( typeof centerX === 'object' ) {
        // arc( center, radius, startAngle, endAngle, anticlockwise )
        anticlockwise = endAngle;
        endAngle = startAngle;
        startAngle = radius;
        radius = centerY;
        var center = centerX;
        this.addPiece( new Piece.Arc( center, radius, startAngle, endAngle, anticlockwise ) );
      } else {
        // arc( centerX, centerY, radius, startAngle, endAngle, anticlockwise )
        this.addPiece( new Piece.Arc( p( centerX, centerY ), radius, startAngle, endAngle, anticlockwise ) );
      }
      return this;
    },
    
    /*
     * Draws an elliptical arc using the Canvas 2D semantics, with the following parameters:
     * ellipticalArc( center, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise )
     * ellipticalArc( centerX, centerY, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise )
     */
    ellipticalArc: function( centerX, centerY, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) {
      // TODO: Ellipse/EllipticalArc has a mess of parameters. Consider parameter object, or double-check parameter handling
      if ( typeof centerX === 'object' ) {
        // ellipticalArc( center, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise )
        anticlockwise = endAngle;
        endAngle = startAngle;
        startAngle = rotation;
        rotation = radiusY;
        radiusY = radiusX;
        radiusX = centerY;
        var center = centerX;
        this.addPiece( new Piece.EllipticalArc( center, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) );
      } else {
        // ellipticalArc( centerX, centerY, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise )
        this.addPiece( new Piece.EllipticalArc( p( centerX, centerY ), radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) );
      }
      return this;
    },
    
    rect: function( x, y, width, height ) {
      this.addPiece( new Piece.Rect( x, y, width, height ) );
      return this;
    },

    //Create a round rectangle. All arguments are number.
    roundRect: function( x, y, width, height, arcw, arch ) {
      var lowX = x + arcw;
      var highX = x + width - arcw;
      var lowY = y + arch;
      var highY = y + height - arch;
      // if ( true ) {
      if ( arcw === arch ) {
        // we can use circular arcs, which have well defined stroked offsets
        this.arc( highX, lowY, arcw, -Math.PI / 2, 0, false )
            .arc( highX, highY, arcw, 0, Math.PI / 2, false )
            .arc( lowX, highY, arcw, Math.PI / 2, Math.PI, false )
            .arc( lowX, lowY, arcw, Math.PI, Math.PI * 3 / 2, false )
            .close();
      } else {
        // we have to resort to elliptical arcs
        this.ellipticalArc( highX, lowY, arcw, arch, 0, -Math.PI / 2, 0, false )
            .ellipticalArc( highX, highY, arcw, arch, 0, 0, Math.PI / 2, false )
            .ellipticalArc( lowX, highY, arcw, arch, 0, Math.PI / 2, Math.PI, false )
            .ellipticalArc( lowX, lowY, arcw, arch, 0, Math.PI, Math.PI * 3 / 2, false )
            .close();
      }
      return this;
    },
    
    close: function() {
      this.addPiece( new Piece.Close() );
      return this;
    },
    
    copy: function() {
      return new Shape( this.pieces );
    },
    
    addPiece: function( piece ) {
      this.pieces.push( piece );
      piece.applyPiece( this );
      this.invalidate();
      assert && assert( this.bounds.isEmpty() || this.bounds.isFinite(), 'shape bounds infinite after adding piece: ' + piece );
      return this; // allow for chaining
    },
    
    // write out this shape's path to a canvas 2d context. does NOT include the beginPath()!
    writeToContext: function( context ) {
      _.each( this.pieces, function( piece ) {
        piece.writeToContext( context );
      } );
    },
    
    // returns something like "M150 0 L75 200 L225 200 Z" for a triangle
    getSVGPath: function() {
      var subpathStrings = [];
      _.each( this.subpaths, function( subpath ) {
        if( subpath.isDrawable() ) {
          // since the commands after this are relative to the previous 'point', we need to specify a move to the initial point
          var startPoint = subpath.getFirstSegment().start;
          assert && assert( startPoint.equals( subpath.getFirstPoint(), 0.00001 ) ); // sanity check
          var string = 'M ' + startPoint.x + ' ' + startPoint.y + ' ';
          
          string += _.map( subpath.segments, function( segment ) { return segment.getSVGPathFragment(); } ).join( ' ' );
          
          if ( subpath.isClosed() ) {
            string += ' Z';
          }
          subpathStrings.push( string );
        }
      } );
      return subpathStrings.join( ' ' );
    },
    
    // return a new Shape that is transformed by the associated matrix
    transformed: function( matrix ) {
      return new Shape( _.flatten( _.map( this.pieces, function( piece ) { return piece.transformed( matrix ); } ), true ) );
    },
    
    // returns the bounds. if lineStyles exists, include the stroke in the bounds
    // TODO: consider renaming to getBounds()?
    computeBounds: function( lineStyles ) {
      if ( lineStyles ) {
        return this.bounds.union( this.getStrokedShape( lineStyles ).bounds );
      } else {
        return this.bounds;
      }
    },
    
    containsPoint: function( point ) {
      // we pick a ray, and determine the winding number over that ray. if the number of segments crossing it CCW == number of segments crossing it CW, then the point is contained in the shape
      var ray = new Ray2( point, p( 1, 0 ) );
      
      return this.windingIntersection( ray ) !== 0;
    },
    
    intersection: function( ray ) {
      var hits = [];
      _.each( this.subpaths, function( subpath ) {
        if ( subpath.isDrawable() ) {
          _.each( subpath.segments, function( segment ) {
            _.each( segment.intersection( ray ), function( hit ) {
              hits.push( hit );
            } );
          } );
          
          if ( subpath.hasClosingSegment() ) {
            _.each( subpath.getClosingSegment().intersection( ray ), function( hit ) {
              hits.push( hit );
            } );
          }
        }
      } );
      return _.sortBy( hits, function( hit ) { return hit.distance; } );
    },
    
    windingIntersection: function( ray ) {
      var wind = 0;
      
      _.each( this.subpaths, function( subpath ) {
        if ( subpath.isDrawable() ) {
          _.each( subpath.segments, function( segment ) {
            wind += segment.windingIntersection( ray );
          } );
          
          // handle the implicit closing line segment
          if ( subpath.hasClosingSegment() ) {
            wind += subpath.getClosingSegment().windingIntersection( ray );
          }
        }
      } );
      
      return wind;
    },
    
    intersectsBounds: function( bounds ) {
      var intersects = false;
      // TODO: break-out-early optimizations
      _.each( this.subpaths, function( subpath ) {
        if ( subpath.isDrawable() ) {
          _.each( subpath.segments, function( segment ) {
            intersects = intersects && segment.intersectsBounds( bounds );
          } );
          
          // handle the implicit closing line segment
          if ( subpath.hasClosingSegment() ) {
            intersects = intersects && subpath.getClosingSegment().intersectsBounds( bounds );
          }
        }
      } );
      return intersects;
    },
    
    invalidate: function() {
      this._strokedShapeComputed = false;
    },
    
    // returns a new Shape that is an outline of the stroked path of this current Shape. currently not intended to be nested (doesn't do intersection computations yet)
    getStrokedShape: function( lineStyles ) {
      
      if ( lineStyles === undefined ) {
        lineStyles = new kite.LineStyles();
      }
      
      // return a cached version if possible
      if ( this._strokedShapeComputed && this._strokedShapeStyles.equals( lineStyles ) ) {
        return this._strokedShape;
      }
      
      // filter out subpaths where nothing would be drawn
      var subpaths = _.filter( this.subpaths, function( subpath ) { return subpath.isDrawable(); } );
      
      var shape = new Shape();
      
      var lineWidth = lineStyles.lineWidth;
      
      // joins two segments together on the logical "left" side, at 'center' (where they meet), and normalized tangent vectors in the direction of the stroking
      // to join on the "right" side, switch the tangent order and negate them
      function join( center, fromTangent, toTangent ) {
        // where our join path starts and ends
        var fromPoint = center.plus( fromTangent.perpendicular().negated().times( lineWidth / 2 ) );
        var toPoint = center.plus( toTangent.perpendicular().negated().times( lineWidth / 2 ) );
        
        // only insert a join on the non-acute-angle side
        if ( fromTangent.perpendicular().dot( toTangent ) > 0 ) {
          switch( lineStyles.lineJoin ) {
            case 'round':
              var fromAngle = fromTangent.angle() + Math.PI / 2;
              var toAngle = toTangent.angle() + Math.PI / 2;
              shape.addPiece( new Piece.Arc( center, lineWidth / 2, fromAngle, toAngle, true ) );
              break;
            case 'miter':
              var theta = fromTangent.angleBetween( toTangent.negated() );
              var notStraight = theta < Math.PI - 0.00001; // if fromTangent is approximately equal to toTangent, just bevel. it will be indistinguishable
              if ( 1 / Math.sin( theta / 2 ) <= lineStyles.miterLimit && theta < Math.PI - 0.00001 ) {
                // draw the miter
                var miterPoint = lineLineIntersection( fromPoint, fromPoint.plus( fromTangent ), toPoint, toPoint.plus( toTangent ) );
                shape.addPiece( new Piece.LineTo( miterPoint ) );
                shape.addPiece( new Piece.LineTo( toPoint ) );
              } else {
                // angle too steep, use bevel instead. same as below, but copied for linter
                shape.addPiece( new Piece.LineTo( toPoint ) );
              }
              break;
            case 'bevel':
              shape.addPiece( new Piece.LineTo( toPoint ) );
              break;
          }
        } else {
          // no join necessary here since we have the acute angle. just simple lineTo for now so that the next segment starts from the right place
          // TODO: can we prevent self-intersection here?
          if ( !fromPoint.equals( toPoint ) ) {
            shape.addPiece( new Piece.LineTo( toPoint ) );
          }
        }
      }
      
      // draws the necessary line cap from the endpoint 'center' in the direction of the tangent
      function cap( center, tangent ) {
        switch( lineStyles.lineCap ) {
          case 'butt':
            shape.addPiece( new Piece.LineTo( center.plus( tangent.perpendicular().times( lineWidth / 2 ) ) ) );
            break;
          case 'round':
            var tangentAngle = tangent.angle();
            shape.addPiece( new Piece.Arc( center, lineWidth / 2, tangentAngle + Math.PI / 2, tangentAngle - Math.PI / 2, true ) );
            break;
          case 'square':
            var toLeft = tangent.perpendicular().negated().times( lineWidth / 2 );
            var toRight = tangent.perpendicular().times( lineWidth / 2 );
            var toFront = tangent.times( lineWidth / 2 );
            shape.addPiece( new Piece.LineTo( center.plus( toLeft ).plus( toFront ) ) );
            shape.addPiece( new Piece.LineTo( center.plus( toRight ).plus( toFront ) ) );
            shape.addPiece( new Piece.LineTo( center.plus( toRight ) ) );
            break;
        }
      }
      
      _.each( subpaths, function( subpath ) {
        var i;
        var segments = subpath.segments;
        
        // TODO: shortcuts for _.first( segments ) and _.last( segments ),
        
        // we don't need to insert an implicit closing segment if the start and end points are the same
        var alreadyClosed = _.last( segments ).end.equals( _.first( segments ).start );
        // if there is an implicit closing segment
        var closingSegment = alreadyClosed ? null : new kite.Segment.Line( segments[segments.length-1].end, segments[0].start );
        
        // move to the first point in our stroked path
        shape.addPiece( new Piece.MoveTo( segmentStartLeft( _.first( segments ), lineWidth ) ) );
        
        // stroke the logical "left" side of our path
        for ( i = 0; i < segments.length; i++ ) {
          if ( i > 0 ) {
            join( segments[i].start, segments[i-1].endTangent, segments[i].startTangent, true );
          }
          _.each( segments[i].strokeLeft( lineWidth ), function( piece ) {
            shape.addPiece( piece );
          } );
        }
        
        // handle the "endpoint"
        if ( subpath.closed ) {
          if ( alreadyClosed ) {
            join( _.last( segments ).end, _.last( segments ).endTangent, _.first( segments ).startTangent );
            shape.addPiece( new Piece.Close() );
            shape.addPiece( new Piece.MoveTo( segmentStartRight( _.first( segments ), lineWidth ) ) );
            join( _.last( segments ).end, _.first( segments ).startTangent.negated(), _.last( segments ).endTangent.negated() );
          } else {
            // logical "left" stroke on the implicit closing segment
            join( closingSegment.start, _.last( segments ).endTangent, closingSegment.startTangent );
            _.each( closingSegment.strokeLeft( lineWidth ), function( piece ) {
              shape.addPiece( piece );
            } );
            
            // TODO: similar here to other block of if.
            join( closingSegment.end, closingSegment.endTangent, _.first( segments ).startTangent );
            shape.addPiece( new Piece.Close() );
            shape.addPiece( new Piece.MoveTo( segmentStartRight( _.first( segments ), lineWidth ) ) );
            join( closingSegment.end, _.first( segments ).startTangent.negated(), closingSegment.endTangent.negated() );
            
            // logical "right" stroke on the implicit closing segment
            _.each( closingSegment.strokeRight( lineWidth ), function( piece ) {
              shape.addPiece( piece );
            } );
            join( closingSegment.start, closingSegment.startTangent.negated(), _.last( segments ).endTangent.negated() );
          }
        } else {
          cap( _.last( segments ).end, _.last( segments ).endTangent );
        }
        
        // stroke the logical "right" side of our path
        for ( i = segments.length - 1; i >= 0; i-- ) {
          if ( i < segments.length - 1 ) {
            join( segments[i].end, segments[i+1].startTangent.negated(), segments[i].endTangent.negated(), false );
          }
          _.each( segments[i].strokeRight( lineWidth ), function( piece ) {
            shape.addPiece( piece );
          } );
        }
        
        // handle the start point
        if ( subpath.closed ) {
          // we already did the joins, just close the 'right' side
          shape.addPiece( new Piece.Close() );
        } else {
          cap( _.first( segments ).start, _.first( segments ).startTangent.negated() );
          shape.addPiece( new Piece.Close() );
        }
      } );
      
      this._strokedShape = shape;
      this._strokedShapeComputed = true;
      this._strokedShapeStyles = new kite.LineStyles( lineStyles ); // shallow copy, since we consider linestyles to be mutable
      
      return shape;
    },
    
    toString: function() {
      var result = 'new kite.Shape()';
      _.each( this.pieces, function( piece ) {
        result += '.' + piece.toString();
      } );
      return result;
    },
    
    /*---------------------------------------------------------------------------*
    * Internal subpath computations
    *----------------------------------------------------------------------------*/
    
    ensure: function( point ) {
      if ( !this.hasSubpaths() ) {
        this.addSubpath( new Subpath() );
        this.getLastSubpath().addPoint( point );
      }
    },
    
    addSubpath: function( subpath ) {
      this.subpaths.push( subpath );
    },
    
    hasSubpaths: function() {
      return this.subpaths.length > 0;
    },
    
    getLastSubpath: function() {
      return _.last( this.subpaths );
    }
  };
  
  /*---------------------------------------------------------------------------*
  * Shape shortcuts
  *----------------------------------------------------------------------------*/
  
  Shape.rectangle = function( x, y, width, height ) {
    return new Shape().rect( x, y, width, height );
  };
  Shape.rect = Shape.rectangle;

  //Create a round rectangle. All arguments are number.
  //Rounding is currently using quadraticCurveTo.  Please note, future versions may use arcTo
  //TODO: rewrite with arcTo?
  Shape.roundRect = function( x, y, width, height, arcw, arch ) {
    return new Shape().roundRect( x, y, width, height, arcw, arch );
  };
  Shape.roundRectangle = Shape.roundRect;
  
  Shape.bounds = function( bounds ) {
    return new Shape().rect( bounds.minX, bounds.minY, bounds.maxX - bounds.minX, bounds.maxY - bounds.minY );
  };

  //Create a line segment, using either (x1,y1,x2,y2) or ({x1,y1},{x2,y2}) arguments
  Shape.lineSegment = function( a, b, c, d ) {
    // TODO: add type assertions?
    if ( typeof a === 'number' ) {
      return new Shape().moveTo( a, b ).lineTo( c, d );
    }
    else {
      return new Shape().moveTo( a ).lineTo( b );
    }
  };
  
  Shape.regularPolygon = function( sides, radius ) {
    var first = true;
    return new Shape( _.map( _.range( sides ), function( k ) {
      var theta = 2 * Math.PI * k / sides;
      if ( first ) {
        first = false;
        // first segment should be a moveTo
        return new Piece.MoveTo( p( radius * Math.cos( theta ), radius * Math.sin( theta ) ) );
      } else {
        return new Piece.LineTo( p( radius * Math.cos( theta ), radius * Math.sin( theta ) ) );
      }
    } ), true );
  };
  
  // supports both circle( centerX, centerY, radius ), circle( center, radius ), and circle( radius ) with the center default to 0,0
  Shape.circle = function( centerX, centerY, radius ) {
    if ( centerY === undefined ) {
      // circle( radius ), center = 0,0
      return new Shape().circle( 0, 0, centerX );
    }
    return new Shape().circle( centerX, centerY, radius ).close();
  };
  
  /*
   * Supports ellipse( centerX, centerY, radiusX, radiusY ), ellipse( center, radiusX, radiusY ), and ellipse( radiusX, radiusY )
   * with the center default to 0,0 and rotation of 0
   */
  Shape.ellipse = function( centerX, centerY, radiusX, radiusY ) {
    // TODO: Ellipse/EllipticalArc has a mess of parameters. Consider parameter object, or double-check parameter handling
    if ( radiusX === undefined ) {
      // ellipse( radiusX, radiusY ), center = 0,0
      return new Shape().ellipse( 0, 0, centerX, centerY );
    }
    return new Shape().ellipse( centerX, centerY, radiusX, radiusY ).close();
  };
  
  // supports both arc( centerX, centerY, radius, startAngle, endAngle, anticlockwise ) and arc( center, radius, startAngle, endAngle, anticlockwise )
  Shape.arc = function( centerX, centerY, radius, startAngle, endAngle, anticlockwise ) {
    return new Shape().arc( centerX, centerY, radius, startAngle, endAngle, anticlockwise );
  };
  
  
  // TODO: performance / cleanliness to have these as methods instead?
  function segmentStartLeft( segment, lineWidth ) {
    assert && assert( lineWidth !== undefined );
    return segment.start.plus( segment.startTangent.perpendicular().negated().times( lineWidth / 2 ) );
  }
  
  function segmentEndLeft( segment, lineWidth ) {
    assert && assert( lineWidth !== undefined );
    return segment.end.plus( segment.endTangent.perpendicular().negated().times( lineWidth / 2 ) );
  }
  
  function segmentStartRight( segment, lineWidth ) {
    assert && assert( lineWidth !== undefined );
    return segment.start.plus( segment.startTangent.perpendicular().times( lineWidth / 2 ) );
  }
  
  function segmentEndRight( segment, lineWidth ) {
    assert && assert( lineWidth !== undefined );
    return segment.end.plus( segment.endTangent.perpendicular().times( lineWidth / 2 ) );
  }
  
  return Shape;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Base code for layers that helps with shared layer functions
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/layers/Layer',['require','ASSERT/assert','ASSERT/assert','DOT/Bounds2','DOT/Transform3','SCENERY/scenery','SCENERY/util/Trail'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  var assertExtra = require( 'ASSERT/assert' )( 'scenery.extra', true );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Transform3 = require( 'DOT/Transform3' );
  
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );
  
  var globalIdCounter = 1;
  
  /*
   * Required arguments:
   * $main     - the jQuery-wrapped container for the scene
   * scene     - the scene itself
   * baseNode  - the base node for this layer
   *
   * Optional arguments:
   * batchDOMChanges: false - Only run DOM manipulation from within requestAnimationFrame calls
   */
  scenery.Layer = function( args ) {
    
    // assign a unique ID to this layer
    this._id = globalIdCounter++;
    
    this.$main = args.$main;
    this.scene = args.scene;
    this.baseNode = args.baseNode;
    
    // DOM batching
    this.batchDOMChanges = args.batchDOMChanges || false;
    this.pendingDOMChanges = [];
    this.applyingDOMChanges = false;
    
    // TODO: cleanup of flags!
    this.usesPartialCSSTransforms = args.cssTranslation || args.cssRotation || args.cssScale;
    this.cssTranslation = args.cssTranslation; // CSS for the translation
    this.cssRotation = args.cssRotation;       // CSS for the rotation
    this.cssScale = args.cssScale;             // CSS for the scaling
    this.cssTransform = args.cssTransform;     // CSS for the entire base node (will ignore other partial transforms)
    assert && assert( !( this.usesPartialCSSTransforms && this.cssTransform ), 'Do not specify both partial and complete CSS transform arguments.' );
    
    // initialize to fully dirty so we draw everything the first time
    // bounds in global coordinate frame
    this.dirtyBounds = Bounds2.EVERYTHING;
    
    this.setStartBoundary( args.startBoundary );
    this.setEndBoundary( args.endBoundary );
    
    // set baseTrail from the scene to our baseNode
    if ( this.baseNode === this.scene ) {
      this.baseTrail = new scenery.Trail( this.scene );
    } else {
      this.baseTrail = this.startPointer.trail.copy();
      assert && assert( this.baseTrail.lastNode() === this.baseNode );
    }
    
    // we reference all trails in an unordered way
    this._layerTrails = [];
    
    var layer = this;
    
    // whenever the base node's children or self change bounds, signal this. we want to explicitly ignore the base node's main bounds for
    // CSS transforms, since the self / children bounds may not have changed
    this.baseNodeListener = {
      selfBounds: function( bounds ) {
        layer.baseNodeInternalBoundsChange();
      },
      
      childBounds: function( bounds ) {
        layer.baseNodeInternalBoundsChange();
      }
    };
    this.baseNode.addEventListener( this.baseNodeListener );
    
    this.fitToBounds = this.usesPartialCSSTransforms || this.cssTransform;
    assert && assert( this.fitToBounds || this.baseNode === this.scene, 'If the baseNode is not the scene, we need to fit the bounds' );
    
    // used for CSS transforms where we need to transform our base node's bounds into the (0,0,w,h) bounds range
    this.baseNodeTransform = new Transform3();
    //this.baseNodeInteralBounds = Bounds2.NOTHING; // stores the bounds transformed into (0,0,w,h)
  };
  var Layer = scenery.Layer;
  
  Layer.prototype = {
    constructor: Layer,
    
    setStartBoundary: function( boundary ) {
      // console.log( 'setting start boundary on layer ' + this.getId() + ': ' + boundary.toString() );
      this.startBoundary = boundary;
      
      // TODO: deprecate these, use boundary references instead? or boundary convenience functions
      this.startPointer = this.startBoundary.nextStartPointer;
      this.startPaintedTrail = this.startBoundary.nextPaintedTrail;
    },
    
    setEndBoundary: function( boundary ) {
      // console.log( 'setting end boundary on layer ' + this.getId() + ': ' + boundary.toString() );
      this.endBoundary = boundary;
      
      // TODO: deprecate these, use boundary references instead? or boundary convenience functions
      this.endPointer = this.endBoundary.previousEndPointer;
      this.endPaintedTrail = this.endBoundary.previousPaintedTrail;
    },
    
    getStartPointer: function() {
      return this.startPointer;
    },
    
    getEndPointer: function() {
      return this.endPointer;
    },
    
    flushDOMChanges: function() {
      // signal that we are now applying the changes, so calling domChange will trigger instant evaluation
      this.applyingDOMChanges = true;
      
      // TODO: consider a 'try' block, as things may now not exist? ideally we should only batch things that will always work
      _.each( this.pendingDOMChanges, function( change ) {
        change();
      } );
      
      // removes all entries
      this.pendingDOMChanges.splice( 0, this.pendingDOMChanges.length );
      
      // start batching again
      this.applyingDOMChanges = false;
    },
    
    domChange: function( callback ) {
      if ( this.batchDOMChanges && !this.applyingDOMChanges ) {
        this.pendingDOMChanges.push( callback );
      } else {
        callback();
      }
    },
    
    toString: function() {
      return this.getName() + ' ' + ( this.startPointer ? this.startPointer.toString() : '!' ) + ' (' + ( this.startPaintedTrail ? this.startPaintedTrail.toString() : '!' ) + ') => ' + ( this.endPointer ? this.endPointer.toString() : '!' ) + ' (' + ( this.endPaintedTrail ? this.endPaintedTrail.toString() : '!' ) + ')';
    },
    
    getId: function() {
      return this._id;
    },
    
    // trails associated with the layer, NOT necessarily in order
    getLayerTrails: function() {
      return this._layerTrails.slice( 0 );
    },
    
    /*---------------------------------------------------------------------------*
    * Abstract
    *----------------------------------------------------------------------------*/
    
    render: function( state ) {
      throw new Error( 'Layer.render unimplemented' );
    },
    
    // TODO: consider a stack-based model for transforms?
    // TODO: is this necessary? verify with the render state
    applyTransformationMatrix: function( matrix ) {
      throw new Error( 'Layer.applyTransformationMatrix unimplemented' );
    },
    
    // adds a trail (with the last node) to the layer
    addNodeFromTrail: function( trail ) {
      // console.log( 'addNodeFromTrail layer: ' + this.getId() + ', trail: ' + trail.toString() );
      // TODO: sync this with DOMLayer's implementation
      this._layerTrails.push( trail );
    },
    
    // removes a trail (with the last node) to the layer
    removeNodeFromTrail: function( trail ) {
      // console.log( 'removeNodeFromTrail layer: ' + this.getId() + ', trail: ' + trail.toString() );
      // TODO: sync this with DOMLayer's implementation
      var i;
      for ( i = 0; i < this._layerTrails.length; i++ ) {
        this._layerTrails[i].reindex();
        if ( this._layerTrails[i].compare( trail ) === 0 ) {
          break;
        }
      }
      assert && assert( i < this._layerTrails.length );
      this._layerTrails.splice( i, 1 );
    },
    
    // returns next zIndex in place. allows layers to take up more than one single zIndex
    reindex: function( zIndex ) {
      throw new Error( 'unimplemented layer reindex' );
    },
    
    pushClipShape: function( shape ) {
      throw new Error( 'Layer.pushClipShape unimplemented' );
    },
    
    popClipShape: function() {
      throw new Error( 'Layer.popClipShape unimplemented' );
    },
    
    renderToCanvas: function( canvas, context, delayCounts ) {
      throw new Error( 'Layer.renderToCanvas unimplemented' );
    },
    
    dispose: function() {
      this.baseNode.removeEventListener( this.baseNodeListener );
    },
    
    // args should contain node, bounds (local bounds), transform, trail
    markDirtyRegion: function( args ) {
      throw new Error( 'Layer.markDirtyRegion unimplemented' );
    },
    
    // args should contain node, type (append, prepend, set), matrix, transform, trail
    transformChange: function( args ) {
      throw new Error( 'Layer.transformChange unimplemented' );
    },
    
    getName: function() {
      throw new Error( 'Layer.getName unimplemented' );
    },
    
    // called when the base node's "internal" (self or child) bounds change, but not when it is just from the base node's own transform changing
    baseNodeInternalBoundsChange: function() {
      // no error, many times this doesn't need to be handled
    }
  };
  
  Layer.cssTransformPadding = 3;
  
  return Layer;
} );



// Copyright 2002-2012, University of Colorado

/**
 * Wraps the context and contains a reference to the canvas, so that we can absorb unnecessary state changes,
 * and possibly combine certain fill operations.
 *
 * TODO: performance analysis, possibly axe this and use direct modification.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/util/CanvasContextWrapper',['require','ASSERT/assert','SCENERY/scenery'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.CanvasContextWrapper = function( canvas, context ) {
    this.canvas = canvas;
    this.context = context;
    
    this.resetStyles();
  };
  var CanvasContextWrapper = scenery.CanvasContextWrapper;
  
  CanvasContextWrapper.prototype = {
    constructor: CanvasContextWrapper,
    
    // set local styles to undefined, so that they will be invalidated later
    resetStyles: function() {
      this.fillStyle = undefined; // null
      this.strokeStyle = undefined; // null
      this.lineWidth = undefined; // 1
      this.lineCap = undefined; // 'butt'
      this.lineJoin = undefined; // 'miter'
      this.lineDash = undefined; // null
      this.lineDashOffset = undefined; // 0
      this.miterLimit = undefined; // 10
      
      this.font = undefined; // '10px sans-serif'
      this.textAlign = undefined; // 'start'
      this.textBaseline = undefined; // 'alphabetic'
      this.direction = undefined; // 'inherit'
    },
    
    setFillStyle: function( style ) {
      if ( this.fillStyle !== style ) {
        this.fillStyle = style;
        
        // allow gradients / patterns
        this.context.fillStyle = ( style && style.getCanvasStyle ) ? style.getCanvasStyle() : style;
      }
    },
    
    setStrokeStyle: function( style ) {
      if ( this.strokeStyle !== style ) {
        this.strokeStyle = style;
        
        // allow gradients / patterns
        this.context.strokeStyle = ( style && style.getCanvasStyle ) ? style.getCanvasStyle() : style;
      }
    },
    
    setLineWidth: function( width ) {
      if ( this.lineWidth !== width ) {
        this.lineWidth = width;
        this.context.lineWidth = width;
      }
    },
    
    setLineCap: function( cap ) {
      if ( this.lineCap !== cap ) {
        this.lineCap = cap;
        this.context.lineCap = cap;
      }
    },
    
    setLineJoin: function( join ) {
      if ( this.lineJoin !== join ) {
        this.lineJoin = join;
        this.context.lineJoin = join;
      }
    },
    
    setLineDash: function( dash ) {
      assert && assert( dash !== undefined, 'undefined line dash would cause hard-to-trace errors' );
      if ( this.lineDash !== dash ) {
        this.lineDash = dash;
        if ( this.context.setLineDash ) {
          this.context.setLineDash( dash );
        } else if ( this.context.mozDash !== undefined ) {
          this.context.mozDash = dash;
        } else {
          // unsupported line dash! do... nothing?
        }
      }
    },
    
    setLineDashOffset: function( lineDashOffset ) {
      if ( this.lineDashOffset !== lineDashOffset ) {
        this.lineDashOffset = lineDashOffset;
        this.context.lineDashOffset = lineDashOffset;
      }
    },
    
    setFont: function( font ) {
      if ( this.font !== font ) {
        this.font = font;
        this.context.font = font;
      }
    },
    
    setTextAlign: function( textAlign ) {
      if ( this.textAlign !== textAlign ) {
        this.textAlign = textAlign;
        this.context.textAlign = textAlign;
      }
    },
    
    setTextBaseline: function( textBaseline ) {
      if ( this.textBaseline !== textBaseline ) {
        this.textBaseline = textBaseline;
        this.context.textBaseline = textBaseline;
      }
    },
    
    setDirection: function( direction ) {
      if ( this.direction !== direction ) {
        this.direction = direction;
        this.context.direction = direction;
      }
    }
  };
  
  return CanvasContextWrapper;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Points to a specific node (with a trail), and whether it is conceptually before or after the node.
 *
 * There are two orderings:
 * - rendering order: the order that node selves would be rendered, matching the Trail implicit order
 * - nesting order:   the order in depth first with entering a node being "before" and exiting a node being "after"
 *
 * TODO: more seamless handling of the orders. or just exclusively use the nesting order
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/util/TrailPointer',['require','ASSERT/assert','SCENERY/scenery','SCENERY/util/Trail'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  require( 'SCENERY/util/Trail' );
  
  /*
   * isBefore: whether this points to before the node (and its children) have been rendered, or after
   */
  scenery.TrailPointer = function( trail, isBefore ) {
    assert && assert( trail instanceof scenery.Trail, 'trail is not a trail' );
    this.trail = trail;
    
    this.setBefore( isBefore );
  };
  var TrailPointer = scenery.TrailPointer;
  
  TrailPointer.prototype = {
    constructor: TrailPointer,
    
    copy: function() {
      return new TrailPointer( this.trail.copy(), this.isBefore );
    },
    
    setBefore: function( isBefore ) {
      this.isBefore = isBefore;
      this.isAfter = !isBefore;
    },
    
    // return the equivalent pointer that swaps before and after (may return null if it doesn't exist)
    getRenderSwappedPointer: function() {
      var newTrail = this.isBefore ? this.trail.previous() : this.trail.next();
      
      if ( newTrail === null ) {
        return null;
      } else {
        return new TrailPointer( newTrail, !this.isBefore );
      }
    },
    
    getRenderBeforePointer: function() {
      return this.isBefore ? this : this.getRenderSwappedPointer();
    },
    
    getRenderAfterPointer: function() {
      return this.isAfter ? this : this.getRenderSwappedPointer();
    },
    
    /*
     * In the render order, will return 0 if the pointers are equivalent, -1 if this pointer is before the
     * other pointer, and 1 if this pointer is after the other pointer.
     */
    compareRender: function( other ) {
      assert && assert( other !== null );
      
      var a = this.getRenderBeforePointer();
      var b = other.getRenderBeforePointer();
      
      if ( a !== null && b !== null ) {
        // normal (non-degenerate) case
        return a.trail.compare( b.trail );
      } else {
        // null "before" point is equivalent to the "after" pointer on the last rendered node.
        if ( a === b ) {
          return 0; // uniqueness guarantees they were the same
        } else {
          return a === null ? 1 : -1;
        }
      }
    },
    
    /*
     * Like compareRender, but for the nested (depth-first) order
     *
     * TODO: optimization?
     */
    compareNested: function( other ) {
      assert && assert( other );
      
      var comparison = this.trail.compare( other.trail );
      
      if ( comparison === 0 ) {
        // if trails are equal, just compare before/after
        if ( this.isBefore === other.isBefore ) {
          return 0;
        } else {
          return this.isBefore ? -1 : 1;
        }
      } else {
        // if one is an extension of the other, the shorter isBefore flag determines the order completely
        if ( this.trail.isExtensionOf( other.trail ) ) {
          return other.isBefore ? 1 : -1;
        } else if ( other.trail.isExtensionOf( this.trail ) ) {
          return this.isBefore ? -1 : 1;
        } else {
          // neither is a subtrail of the other, so a straight trail comparison should give the answer
          return comparison;
        }
      }
    },
    
    equalsRender: function( other ) {
      return this.compareRender( other ) === 0;
    },
    
    equalsNested: function( other ) {
      return this.compareNested( other ) === 0;
    },
    
    // will return false if this pointer has gone off of the beginning or end of the tree (will be marked with isAfter or isBefore though)
    hasTrail: function() {
      return !!this.trail;
    },
    
    // TODO: refactor with "Side"-like handling
    // moves this pointer forwards one step in the nested order
    nestedForwards: function() {
      if ( this.isBefore ) {
        if ( this.trail.lastNode()._children.length > 0 ) {
          // stay as before, just walk to the first child
          this.trail.addDescendant( this.trail.lastNode()._children[0], 0 );
        } else {
          // stay on the same node, but switch to after
          this.setBefore( false );
        }
      } else {
        if ( this.trail.indices.length === 0 ) {
          // nothing else to jump to below, so indicate the lack of existence
          this.trail = null;
          // stays isAfter
          return null;
        } else {
          var index = this.trail.indices[this.trail.indices.length - 1];
          this.trail.removeDescendant();
          
          if ( this.trail.lastNode()._children.length > index + 1 ) {
            // more siblings, switch to the beginning of the next one
            this.trail.addDescendant( this.trail.lastNode()._children[index+1], index + 1 );
            this.setBefore( true );
          } else {
            // no more siblings. exit on parent. nothing else needed since we're already isAfter
          }
        }
      }
      return this;
    },
    
    // moves this pointer backwards one step in the nested order
    nestedBackwards: function() {
      if ( this.isBefore ) {
        if ( this.trail.indices.length === 0 ) {
          // jumping off the front
          this.trail = null;
          // stays isBefore
          return null;
        } else {
          var index = this.trail.indices[this.trail.indices.length - 1];
          this.trail.removeDescendant();
          
          if ( index - 1 >= 0 ) {
            // more siblings, switch to the beginning of the previous one and switch to isAfter
            this.trail.addDescendant( this.trail.lastNode()._children[index-1], index - 1 );
            this.setBefore( false );
          } else {
            // no more siblings. enter on parent. nothing else needed since we're already isBefore
          }
        }
      } else {
        if ( this.trail.lastNode()._children.length > 0 ) {
          // stay isAfter, but walk to the last child
          var children = this.trail.lastNode()._children;
          this.trail.addDescendant( children[children.length-1], children.length - 1 );
        } else {
          // switch to isBefore, since this is a leaf node
          this.setBefore( true );
        }
      }
      return this;
    },
    
    // treats the pointer as render-ordered (includes the start pointer 'before' if applicable, excludes the end pointer 'before' if applicable
    eachNodeBetween: function( other, callback ) {
      this.eachTrailBetween( other, function( trail ) {
        callback( trail.lastNode() );
      } );
    },
    
    // treats the pointer as render-ordered (includes the start pointer 'before' if applicable, excludes the end pointer 'before' if applicable
    eachTrailBetween: function( other, callback ) {
      // this should trigger on all pointers that have the 'before' flag, except a pointer equal to 'other'.
      
      // since we exclude endpoints in the depthFirstUntil call, we need to fire this off first
      if ( this.isBefore ) {
        callback( this.trail );
      }
      
      this.depthFirstUntil( other, function( pointer ) {
        if ( pointer.isBefore ) {
          callback( pointer.trail );
        }
      }, true ); // exclude the endpoints so we can ignore the ending 'before' case
    },
    
    /*
     * Recursively (depth-first) iterates over all pointers between this pointer and 'other', calling
     * callback( pointer ) for each pointer. If excludeEndpoints is truthy, the callback will not be
     * called if pointer is equivalent to this pointer or 'other'.
     *
     * If the callback returns a truthy value, the subtree for the current pointer will be skipped
     * (applies only to before-pointers)
     */
    depthFirstUntil: function( other, callback, excludeEndpoints ) {
      // make sure this pointer is before the other, but allow start === end if we are not excluding endpoints
      assert && assert( this.compareNested( other ) <= ( excludeEndpoints ? -1 : 0 ), 'TrailPointer.depthFirstUntil pointers out of order, possibly in both meanings of the phrase!' );
      assert && assert( this.trail.rootNode() === other.trail.rootNode(), 'TrailPointer.depthFirstUntil takes pointers with the same root' );
      
      // sanity check TODO: remove later
      this.trail.reindex();
      other.trail.reindex();
      
      var pointer = this.copy();
      
      var first = true;
      
      while ( !pointer.equalsNested( other ) ) {
        assert && assert( pointer.compareNested( other ) !== 1, 'skipped in depthFirstUntil' );
        var skipSubtree = false;
        
        if ( first ) {
          // start point
          if ( !excludeEndpoints ) {
            skipSubtree = callback( pointer );
          }
          first = false;
        } else {
          // between point
          skipSubtree = callback( pointer );
        }
        
        if ( skipSubtree && pointer.isBefore ) {
          // to skip the subtree, we just change to isAfter
          pointer.setBefore( false );
          
          // if we skip a subtree, make sure we don't run past the ending pointer
          if ( pointer.compareNested( other ) === 1 ) {
            break;
          }
        } else {
          pointer.nestedForwards();
        }
      }
      
      // end point
      if ( !excludeEndpoints ) {
        callback( pointer );
      }
    },
    
    toString: function() {
      return '[' + ( this.isBefore ? 'before' : 'after' ) + ' ' + this.trail.toString().slice( 1 );
    }
  };
  
  return TrailPointer;
} );


// Copyright 2002-2012, University of Colorado

/**
 * General utility functions for Scenery
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/util/Util',['require','ASSERT/assert','ASSERT/assert','SCENERY/scenery','DOT/Matrix3','DOT/Transform3','DOT/Bounds2','DOT/Vector2'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  var assertExtra = require( 'ASSERT/assert' )( 'scenery.extra', true );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Matrix3 = require( 'DOT/Matrix3' );
  var Transform3 = require( 'DOT/Transform3' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Vector2 = require( 'DOT/Vector2' );
  
  // convenience function
  function p( x, y ) {
    return new Vector2( x, y );
  }
  
  // TODO: remove flag and tests after we're done
  var debugChromeBoundsScanning = false;
  
  scenery.Util = {
    // like _.extend, but with hardcoded support for https://github.com/documentcloud/underscore/pull/986
    extend: function( obj ) {
      _.each( Array.prototype.slice.call( arguments, 1 ), function( source ) {
        if ( source ) {
          for ( var prop in source ) {
            Object.defineProperty( obj, prop, Object.getOwnPropertyDescriptor( source, prop ) );
          }
        }
      });
      return obj;
    },
    
    // Object.create polyfill
    objectCreate: Object.create || function ( o ) {
      if ( arguments.length > 1 ) {
        throw new Error( 'Object.create implementation only accepts the first parameter.' );
      }
      function F() {}

      F.prototype = o;
      return new F();
    },
    
    testAssert: function() {
      return 'assert.scenery: ' + ( assert ? 'true' : 'false' );
    },
    
    testAssertExtra: function() {
      return 'assert.scenery.extra: ' + ( assertExtra ? 'true' : 'false' );
    },
    
    /*---------------------------------------------------------------------------*
     * window.requestAnimationFrame polyfill, by Erik Moller (http://my.opera.com/emoller/blog/2011/12/20/requestanimationframe-for-smart-er-animating)
     * referenced by initial Paul Irish article at http://paulirish.com/2011/requestanimationframe-for-smart-animating/
     *----------------------------------------------------------------------------*/
    polyfillRequestAnimationFrame: function() {
      var lastTime = 0;
      var vendors = [ 'ms', 'moz', 'webkit', 'o' ];
      for ( var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x ) {
        window.requestAnimationFrame = window[vendors[x]+'RequestAnimationFrame'];
        window.cancelAnimationFrame = window[vendors[x]+'CancelAnimationFrame'] || window[vendors[x]+'CancelRequestAnimationFrame'];
      }
     
      if ( !window.requestAnimationFrame ) {
        window.requestAnimationFrame = function(callback) {
          var currTime = new Date().getTime();
          var timeToCall = Math.max(0, 16 - (currTime - lastTime));
          var id = window.setTimeout(function() { callback(currTime + timeToCall); },
            timeToCall);
          lastTime = currTime + timeToCall;
          return id;
        };
      }
     
      if ( !window.cancelAnimationFrame ) {
        window.cancelAnimationFrame = function(id) {
          clearTimeout(id);
        };
      }
    },
    
    backingStorePixelRatio: function( context ) {
      return context.webkitBackingStorePixelRatio ||
             context.mozBackingStorePixelRatio ||
             context.msBackingStorePixelRatio ||
             context.oBackingStorePixelRatio ||
             context.backingStorePixelRatio || 1;
    },
    
    // see http://developer.apple.com/library/safari/#documentation/AudioVideo/Conceptual/HTML-canvas-guide/SettingUptheCanvas/SettingUptheCanvas.html#//apple_ref/doc/uid/TP40010542-CH2-SW5
    // and updated based on http://www.html5rocks.com/en/tutorials/canvas/hidpi/
    backingScale: function ( context ) {
      if ( 'devicePixelRatio' in window ) {
        var backingStoreRatio = Util.backingStorePixelRatio( context );
        
        return window.devicePixelRatio / backingStoreRatio;
      }
      return 1;
    },
    
    // given a data snapshot and transform, calculate range on how large / small the bounds can be
    // very conservative, with an effective 1px extra range to allow for differences in anti-aliasing
    // for performance concerns, this does not support skews / rotations / anything but translation and scaling
    scanBounds: function( imageData, resolution, transform ) {
      
      // entry will be true if any pixel with the given x or y value is non-rgba(0,0,0,0)
      var dirtyX = _.map( _.range( resolution ), function() { return false; } );
      var dirtyY = _.map( _.range( resolution ), function() { return false; } );
      
      for ( var x = 0; x < resolution; x++ ) {
        for ( var y = 0; y < resolution; y++ ) {
          var offset = 4 * ( y * resolution + x );
          if ( imageData.data[offset] !== 0 || imageData.data[offset+1] !== 0 || imageData.data[offset+2] !== 0 || imageData.data[offset+3] !== 0 ) {
            dirtyX[x] = true;
            dirtyY[y] = true;
          }
        }
      }
      
      var minX = _.indexOf( dirtyX, true );
      var maxX = _.lastIndexOf( dirtyX, true );
      var minY = _.indexOf( dirtyY, true );
      var maxY = _.lastIndexOf( dirtyY, true );
      
      // based on pixel boundaries. for minBounds, the inner edge of the dirty pixel. for maxBounds, the outer edge of the adjacent non-dirty pixel
      // results in a spread of 2 for the identity transform (or any translated form)
      var extraSpread = resolution / 16; // is Chrome antialiasing really like this? dear god... TODO!!!
      return {
        minBounds: new Bounds2(
          ( minX < 1 || minX >= resolution - 1 ) ? Number.POSITIVE_INFINITY : transform.inversePosition2( p( minX + 1 + extraSpread, 0 ) ).x,
          ( minY < 1 || minY >= resolution - 1 ) ? Number.POSITIVE_INFINITY : transform.inversePosition2( p( 0, minY + 1 + extraSpread ) ).y,
          ( maxX < 1 || maxX >= resolution - 1 ) ? Number.NEGATIVE_INFINITY : transform.inversePosition2( p( maxX - extraSpread, 0 ) ).x,
          ( maxY < 1 || maxY >= resolution - 1 ) ? Number.NEGATIVE_INFINITY : transform.inversePosition2( p( 0, maxY - extraSpread ) ).y
        ),
        maxBounds: new Bounds2(
          ( minX < 1 || minX >= resolution - 1 ) ? Number.NEGATIVE_INFINITY : transform.inversePosition2( p( minX - 1 - extraSpread, 0 ) ).x,
          ( minY < 1 || minY >= resolution - 1 ) ? Number.NEGATIVE_INFINITY : transform.inversePosition2( p( 0, minY - 1 - extraSpread ) ).y,
          ( maxX < 1 || maxX >= resolution - 1 ) ? Number.POSITIVE_INFINITY : transform.inversePosition2( p( maxX + 2 + extraSpread, 0 ) ).x,
          ( maxY < 1 || maxY >= resolution - 1 ) ? Number.POSITIVE_INFINITY : transform.inversePosition2( p( 0, maxY + 2 + extraSpread ) ).y
        )
      };
    },
    
    canvasAccurateBounds: function( renderToContext, options ) {
      // how close to the actual bounds do we need to be?
      var precision = ( options && options.precision ) ? options.precision : 0.001;
      
      // 512x512 default square resolution
      var resolution = ( options && options.resolution ) ? options.resolution : 128;
      
      // at 1/16x default, we want to be able to get the bounds accurately for something as large as 16x our initial resolution
      // divisible by 2 so hopefully we avoid more quirks from Canvas rendering engines
      var initialScale = ( options && options.initialScale ) ? options.initialScale : (1/16);
      
      var minBounds = Bounds2.NOTHING;
      var maxBounds = Bounds2.EVERYTHING;
      
      var canvas = document.createElement( 'canvas' );
      canvas.width = resolution;
      canvas.height = resolution;
      var context = canvas.getContext( '2d' );
      
      if ( debugChromeBoundsScanning ) {
        $( window ).ready( function() {
          var header = document.createElement( 'h2' );
          $( header ).text( 'Bounds Scan' );
          $( '#display' ).append( header );
        } );
      }
      
      function scan( transform ) {
        // save/restore, in case the render tries to do any funny stuff like clipping, etc.
        context.save();
        transform.matrix.canvasSetTransform( context );
        renderToContext( context );
        context.restore();
        
        var data = context.getImageData( 0, 0, resolution, resolution );
        var minMaxBounds = Util.scanBounds( data, resolution, transform );
        
        function snapshotToCanvas( snapshot ) {
            var canvas = document.createElement( 'canvas' );
            canvas.width = resolution;
            canvas.height = resolution;
            var context = canvas.getContext( '2d' );
            context.putImageData( snapshot, 0, 0 );
            $( canvas ).css( 'border', '1px solid black' );
            $( window ).ready( function() {
              //$( '#display' ).append( $( document.createElement( 'div' ) ).text( 'Bounds: ' +  ) );
              $( '#display' ).append( canvas );
            } );
          }
        
        // TODO: remove after debug
        if ( debugChromeBoundsScanning ) {
          snapshotToCanvas( data );
        }
        
        context.clearRect( 0, 0, resolution, resolution );
        
        return minMaxBounds;
      }
      
      // attempts to map the bounds specified to the entire testing canvas (minus a fine border), so we can nail down the location quickly
      function idealTransform( bounds ) {
        // so that the bounds-edge doesn't land squarely on the boundary
        var borderSize = 2;
        
        var scaleX = ( resolution - borderSize * 2 ) / ( bounds.maxX - bounds.minX );
        var scaleY = ( resolution - borderSize * 2 ) / ( bounds.maxY - bounds.minY );
        var translationX = -scaleX * bounds.minX + borderSize;
        var translationY = -scaleY * bounds.minY + borderSize;
        
        return new Transform3( Matrix3.translation( translationX, translationY ).timesMatrix( Matrix3.scaling( scaleX, scaleY ) ) );
      }
      
      var initialTransform = new Transform3( );
      // make sure to initially center our object, so we don't miss the bounds
      initialTransform.append( Matrix3.translation( resolution / 2, resolution / 2 ) );
      initialTransform.append( Matrix3.scaling( initialScale ) );
      
      var coarseBounds = scan( initialTransform );
      
      minBounds = minBounds.union( coarseBounds.minBounds );
      maxBounds = maxBounds.intersection( coarseBounds.maxBounds );
      
      var tempMin, tempMax, refinedBounds;
      
      // minX
      tempMin = maxBounds.minY;
      tempMax = maxBounds.maxY;
      while ( isFinite( minBounds.minX ) && isFinite( maxBounds.minX ) && Math.abs( minBounds.minX - maxBounds.minX ) > precision ) {
        // use maximum bounds except for the x direction, so we don't miss things that we are looking for
        refinedBounds = scan( idealTransform( new Bounds2( maxBounds.minX, tempMin, minBounds.minX, tempMax ) ) );
        
        if ( minBounds.minX <= refinedBounds.minBounds.minX && maxBounds.minX >= refinedBounds.maxBounds.minX ) {
          // sanity check - break out of an infinite loop!
          if ( debugChromeBoundsScanning ) {
            console.log( 'warning, exiting infinite loop!' );
            console.log( 'transformed "min" minX: ' + idealTransform( new Bounds2( maxBounds.minX, maxBounds.minY, minBounds.minX, maxBounds.maxY ) ).transformPosition2( p( minBounds.minX, 0 ) ) );
            console.log( 'transformed "max" minX: ' + idealTransform( new Bounds2( maxBounds.minX, maxBounds.minY, minBounds.minX, maxBounds.maxY ) ).transformPosition2( p( maxBounds.minX, 0 ) ) );
          }
          break;
        }
        
        minBounds = minBounds.withMinX( Math.min( minBounds.minX, refinedBounds.minBounds.minX ) );
        maxBounds = maxBounds.withMinX( Math.max( maxBounds.minX, refinedBounds.maxBounds.minX ) );
        tempMin = Math.max( tempMin, refinedBounds.maxBounds.minY );
        tempMax = Math.min( tempMax, refinedBounds.maxBounds.maxY );
      }
      
      // maxX
      tempMin = maxBounds.minY;
      tempMax = maxBounds.maxY;
      while ( isFinite( minBounds.maxX ) && isFinite( maxBounds.maxX ) && Math.abs( minBounds.maxX - maxBounds.maxX ) > precision ) {
        // use maximum bounds except for the x direction, so we don't miss things that we are looking for
        refinedBounds = scan( idealTransform( new Bounds2( minBounds.maxX, tempMin, maxBounds.maxX, tempMax ) ) );
        
        if ( minBounds.maxX >= refinedBounds.minBounds.maxX && maxBounds.maxX <= refinedBounds.maxBounds.maxX ) {
          // sanity check - break out of an infinite loop!
          if ( debugChromeBoundsScanning ) {
            console.log( 'warning, exiting infinite loop!' );
          }
          break;
        }
        
        minBounds = minBounds.withMaxX( Math.max( minBounds.maxX, refinedBounds.minBounds.maxX ) );
        maxBounds = maxBounds.withMaxX( Math.min( maxBounds.maxX, refinedBounds.maxBounds.maxX ) );
        tempMin = Math.max( tempMin, refinedBounds.maxBounds.minY );
        tempMax = Math.min( tempMax, refinedBounds.maxBounds.maxY );
      }
      
      // minY
      tempMin = maxBounds.minX;
      tempMax = maxBounds.maxX;
      while ( isFinite( minBounds.minY ) && isFinite( maxBounds.minY ) && Math.abs( minBounds.minY - maxBounds.minY ) > precision ) {
        // use maximum bounds except for the y direction, so we don't miss things that we are looking for
        refinedBounds = scan( idealTransform( new Bounds2( tempMin, maxBounds.minY, tempMax, minBounds.minY ) ) );
        
        if ( minBounds.minY <= refinedBounds.minBounds.minY && maxBounds.minY >= refinedBounds.maxBounds.minY ) {
          // sanity check - break out of an infinite loop!
          if ( debugChromeBoundsScanning ) {
            console.log( 'warning, exiting infinite loop!' );
          }
          break;
        }
        
        minBounds = minBounds.withMinY( Math.min( minBounds.minY, refinedBounds.minBounds.minY ) );
        maxBounds = maxBounds.withMinY( Math.max( maxBounds.minY, refinedBounds.maxBounds.minY ) );
        tempMin = Math.max( tempMin, refinedBounds.maxBounds.minX );
        tempMax = Math.min( tempMax, refinedBounds.maxBounds.maxX );
      }
      
      // maxY
      tempMin = maxBounds.minX;
      tempMax = maxBounds.maxX;
      while ( isFinite( minBounds.maxY ) && isFinite( maxBounds.maxY ) && Math.abs( minBounds.maxY - maxBounds.maxY ) > precision ) {
        // use maximum bounds except for the y direction, so we don't miss things that we are looking for
        refinedBounds = scan( idealTransform( new Bounds2( tempMin, minBounds.maxY, tempMax, maxBounds.maxY ) ) );
        
        if ( minBounds.maxY >= refinedBounds.minBounds.maxY && maxBounds.maxY <= refinedBounds.maxBounds.maxY ) {
          // sanity check - break out of an infinite loop!
          if ( debugChromeBoundsScanning ) {
            console.log( 'warning, exiting infinite loop!' );
          }
          break;
        }
        
        minBounds = minBounds.withMaxY( Math.max( minBounds.maxY, refinedBounds.minBounds.maxY ) );
        maxBounds = maxBounds.withMaxY( Math.min( maxBounds.maxY, refinedBounds.maxBounds.maxY ) );
        tempMin = Math.max( tempMin, refinedBounds.maxBounds.minX );
        tempMax = Math.min( tempMax, refinedBounds.maxBounds.maxX );
      }
      
      if ( debugChromeBoundsScanning ) {
        console.log( 'minBounds: ' + minBounds );
        console.log( 'maxBounds: ' + maxBounds );
      }
      
      var result = new Bounds2(
        ( minBounds.minX + maxBounds.minX ) / 2,
        ( minBounds.minY + maxBounds.minY ) / 2,
        ( minBounds.maxX + maxBounds.maxX ) / 2,
        ( minBounds.maxY + maxBounds.maxY ) / 2
      );
      
      // extra data about our bounds
      result.minBounds = minBounds;
      result.maxBounds = maxBounds;
      result.isConsistent = maxBounds.containsBounds( minBounds );
      result.precision = Math.max(
        Math.abs( minBounds.minX - maxBounds.minX ),
        Math.abs( minBounds.minY - maxBounds.minY ),
        Math.abs( minBounds.maxX - maxBounds.maxX ),
        Math.abs( minBounds.maxY - maxBounds.maxY )
      );
      
      // return the average
      return result;
    }
  };
  var Util = scenery.Util;
  
  return Util;
} );

// Copyright 2002-2012, University of Colorado

/**
 * A Canvas-backed layer in the scene graph. Each layer handles dirty-region handling separately,
 * and corresponds to a single canvas / svg element / DOM element in the main container.
 * Importantly, it does not contain rendered content from a subtree of the main
 * scene graph. It only will render a contiguous block of nodes visited in a depth-first
 * manner.
 *
 * Backing store pixel ratio info: http://www.html5rocks.com/en/tutorials/canvas/hidpi/
 *
 * TODO: update internal documentation
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/layers/CanvasLayer',['require','ASSERT/assert','DOT/Bounds2','SCENERY/scenery','KITE/Shape','SCENERY/layers/Layer','SCENERY/util/CanvasContextWrapper','SCENERY/util/Trail','SCENERY/util/TrailPointer','SCENERY/util/Util'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Shape = require( 'KITE/Shape' );
  
  var Layer = require( 'SCENERY/layers/Layer' ); // uses Layer's prototype for inheritance
  require( 'SCENERY/util/CanvasContextWrapper' );
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/util/TrailPointer' );
  require( 'SCENERY/util/Util' );
  
  // assumes main is wrapped with JQuery
  /*
   *
   */
  scenery.CanvasLayer = function( args ) {
    Layer.call( this, args );
    
    // TODO: deprecate Scene's backing scale, and handle this on a layer-by-layer option?
    this.backingScale = args.scene.backingScale;
    if ( args.fullResolution !== undefined ) {
      this.backingScale = args.fullResolution ? scenery.Util.backingScale( document.createElement( 'canvas' ).getContext( '2d' ) ) : 1;
    }
    
    this.logicalWidth = this.$main.width();
    this.logicalHeight = this.$main.height();
    
    var canvas = document.createElement( 'canvas' );
    canvas.width = this.logicalWidth * this.backingScale;
    canvas.height = this.logicalHeight * this.backingScale;
    $( canvas ).css( 'width', this.logicalWidth );
    $( canvas ).css( 'height', this.logicalHeight );
    $( canvas ).css( 'position', 'absolute' );
    
    // add this layer on top (importantly, the constructors of the layers are called in order)
    this.$main.append( canvas );
    
    this.canvas = canvas;
    // this.context = new DebugContext( canvas.getContext( '2d' ) );
    this.context = canvas.getContext( '2d' );
    this.scene = args.scene;
    
    // workaround for Chrome (WebKit) miterLimit bug: https://bugs.webkit.org/show_bug.cgi?id=108763
    this.context.miterLimit = 20;
    this.context.miterLimit = 10;
    
    this.isCanvasLayer = true;
    
    this.wrapper = new scenery.CanvasContextWrapper( this.canvas, this.context );
  };
  var CanvasLayer = scenery.CanvasLayer;
  
  CanvasLayer.prototype = _.extend( {}, Layer.prototype, {
    constructor: CanvasLayer,
    
    /*
     * Renders the canvas layer from the scene
     *
     * Supported args: {
     *   fullRender: true, // disables drawing to just dirty rectangles
     *   TODO: pruning with bounds and flag to disable
     * }
     */
    render: function( scene, args ) {
      args = args || {};
      
      // bail out quickly if possible
      if ( !args.fullRender && this.dirtyBounds.isEmpty() ) {
        return;
      }
      
      // switch to an identity transform
      this.context.setTransform( this.backingScale, 0, 0, this.backingScale, 0, 0 );
      
      var visibleDirtyBounds = args.fullRender ? scene.sceneBounds : this.dirtyBounds.intersection( scene.sceneBounds );
      
      if ( !visibleDirtyBounds.isEmpty() ) {
        this.clearGlobalBounds( visibleDirtyBounds );
        
        if ( !args.fullRender ) {
          this.pushClipShape( Shape.bounds( visibleDirtyBounds ) );
        }
        
        // dirty bounds (clear, possibly set restricted bounds and handling for that)
        // visibility checks
        this.recursiveRender( scene, args );
        
        // exists for now so that we pop the necessary context state
        if ( !args.fullRender ) {
          this.popClipShape();
        }
      }
      
      // we rendered everything, no more dirty bounds
      this.dirtyBounds = Bounds2.NOTHING;
    },
    
    recursiveRender: function( scene, args ) {
      var layer = this;
      var i;
      var startPointer = new scenery.TrailPointer( this.startPaintedTrail, true );
      var endPointer = new scenery.TrailPointer( this.endPaintedTrail, true );
      
      // stack for canvases that need to be painted, since some effects require scratch canvases
      var wrapperStack = [ this.wrapper ]; // type {CanvasContextWrapper}
      this.wrapper.resetStyles(); // let's be defensive, save() and restore() may have been called previously
      
      function requiresScratchCanvas( trail ) {
        return trail.lastNode().getOpacity() < 1;
      }
      
      function getCanvasWrapper() {
        // TODO: verify that this works with hi-def canvases
        // TODO: use a cache of scratch canvases/contexts on the scene for this purpose, instead of creation
        var canvas = document.createElement( 'canvas' );
        canvas.width = layer.logicalWidth * layer.backingScale;
        canvas.height = layer.logicalHeight * layer.backingScale;
        // $( canvas ).css( 'width', layer.logicalWidth );
        // $( canvas ).css( 'height', layer.logicalHeight );
        var context = canvas.getContext( '2d' );
        
        return new scenery.CanvasContextWrapper( canvas, context );
      }
      
      function topWrapper() {
        return wrapperStack[wrapperStack.length-1];
      }
      
      function enter( trail ) {
        var node = trail.lastNode();
        
        if ( requiresScratchCanvas( trail ) ) {
          var wrapper = getCanvasWrapper();
          wrapperStack.push( wrapper );
          
          var newContext = wrapper.context;
          
          // switch to an identity transform
          newContext.setTransform( layer.backingScale, 0, 0, layer.backingScale, 0, 0 );
          
          // properly set the necessary transform on the context
          _.each( trail.nodes, function( node ) {
            node.transform.getMatrix().canvasAppendTransform( newContext );
          } );
        } else {
          node.transform.getMatrix().canvasAppendTransform( topWrapper().context );
        }
        
        if ( node._clipShape ) {
          // TODO: move to wrapper-specific part
          layer.pushClipShape( node._clipShape );
        }
      }
      
      function exit( trail ) {
        var node = trail.lastNode();
        
        if ( node._clipShape ) {
          // TODO: move to wrapper-specific part
          layer.popClipShape();
        }
        
        if ( requiresScratchCanvas( trail ) ) {
          var baseContext = wrapperStack[wrapperStack.length-2].context;
          var topCanvas = wrapperStack[wrapperStack.length-1].canvas;
          
          // apply necessary style transforms before painting our popped canvas onto the next canvas
          var opacityChange = trail.lastNode().getOpacity() < 1;
          if ( opacityChange ) {
            baseContext.globalAlpha = trail.lastNode().getOpacity();
          }
          
          // paint our canvas onto the level below with a straight transform
          baseContext.save();
          baseContext.setTransform( 1, 0, 0, 1, 0, 0 );
          baseContext.drawImage( topCanvas, 0, 0 );
          baseContext.restore();
          
          // reset styles
          if ( opacityChange ) {
            baseContext.globalAlpha = 1;
          }
          
          wrapperStack.pop();
        } else {
          node.transform.getInverse().canvasAppendTransform( topWrapper().context );
        }
      }
      
      /*
       * We count how many invisible nodes are in our trail, so we can properly iterate without inspecting everything.
       * Additionally, state changes (enter/exit) are only done when nodes are visible, so we skip overhead. If
       * invisibleCount > 0, then the current node is invisible.
       */
      var invisibleCount = 0;
      
      var boundaryTrail;
      
      // sanity check, and allows us to get faster speed
      startPointer.trail.reindex();
      endPointer.trail.reindex();
      
      // first, we need to walk the state up to before our pointer (as far as the recursive handling is concerned)
      // if the pointer is 'before' the node, don't call its enterState since this will be taken care of as the first step.
      // if the pointer is 'after' the node, call enterState since it will call exitState immediately inside the loop
      var startWalkLength = startPointer.trail.length - ( startPointer.isBefore ? 1 : 0 );
      boundaryTrail = new scenery.Trail();
      for ( i = 0; i < startWalkLength; i++ ) {
        var startNode = startPointer.trail.nodes[i];
        boundaryTrail.addDescendant( startNode );
        invisibleCount += startNode.isVisible() ? 0 : 1;
        
        if ( invisibleCount === 0 ) {
          // walk up initial state
          enter( boundaryTrail );
        }
      }
      
      startPointer.depthFirstUntil( endPointer, function renderPointer( pointer ) {
        // handle render here
        
        var node = pointer.trail.lastNode();
        
        if ( pointer.isBefore ) {
          invisibleCount += node.isVisible() ? 0 : 1;
          
          if ( invisibleCount === 0 ) {
            enter( pointer.trail );
            
            if ( node.isPainted() ) {
              var wrapper = wrapperStack[wrapperStack.length-1];
              
              // TODO: consider just passing the wrapper. state not needed (for now), context easily accessible
              node.paintCanvas( wrapper );
            }
            
            // TODO: restricted bounds rendering, and possibly generalize depthFirstUntil
            // var children = node._children;
            
            // check if we need to filter the children we render, and ignore nodes with few children (but allow 2, since that may prevent branches)
            // if ( state.childRestrictedBounds && children.length > 1 ) {
            //   var localRestrictedBounds = node.globalToLocalBounds( state.childRestrictedBounds );
              
            //   // don't filter if every child is inside the bounds
            //   if ( !localRestrictedBounds.containsBounds( node.parentToLocalBounds( node._bounds ) ) ) {
            //     children = node.getChildrenWithinBounds( localRestrictedBounds );
            //   }
            // }
            
            // _.each( children, function( child ) {
            //   fullRender( child, state );
            // } );
          } else {
            // not visible, so don't render the entire subtree
            return true;
          }
        } else {
          if ( invisibleCount === 0 ) {
            exit( pointer.trail );
          }
          
          invisibleCount -= node.isVisible() ? 0 : 1;
        }
        
      }, false ); // include endpoints (for now)
      
      // then walk the state back so we don't muck up any context saving that is going on, similar to how we walked it at the start
      // if the pointer is 'before' the node, call exitState since it called enterState inside the loop on it
      // if the pointer is 'after' the node, don't call its exitState since this was already done
      boundaryTrail = endPointer.trail.copy();
      var endWalkLength = endPointer.trail.length - ( endPointer.isAfter ? 1 : 0 );
      for ( i = endWalkLength - 1; i >= 0; i-- ) {
        var endNode = endPointer.trail.nodes[i];
        invisibleCount -= endNode.isVisible() ? 0 : 1;
        
        if ( invisibleCount === 0 ) {
          // walk back the state
          exit( boundaryTrail );
        }
        
        boundaryTrail.removeDescendant();
      }
    },
    
    dispose: function() {
      Layer.prototype.dispose.call( this );
      $( this.canvas ).detach();
    },
    
    // TODO: consider a stack-based model for transforms?
    applyTransformationMatrix: function( matrix ) {
      matrix.canvasAppendTransform( this.context );
    },
    
    // returns next zIndex in place. allows layers to take up more than one single zIndex
    reindex: function( zIndex ) {
      $( this.canvas ).css( 'z-index', zIndex );
      this.zIndex = zIndex;
      return zIndex + 1;
    },
    
    pushClipShape: function( shape ) {
      // store the current state, since browser support for context.resetClip() is not yet in the stable browser versions
      this.context.save();
      
      this.writeClipShape( shape );
    },
    
    popClipShape: function() {
      this.context.restore();
    },
    
    // canvas-specific
    writeClipShape: function( shape ) {
      // set up the clipping
      this.context.beginPath();
      shape.writeToContext( this.context );
      this.context.clip();
    },
    
    clearGlobalBounds: function( bounds ) {
      if ( !bounds.isEmpty() ) {
        this.context.save();
        this.context.setTransform( this.backingScale, 0, 0, this.backingScale, 0, 0 );
        this.context.clearRect( bounds.getX(), bounds.getY(), bounds.getWidth(), bounds.getHeight() );
        // use this for debugging cleared (dirty) regions for now
        // this.context.fillStyle = '#' + Math.floor( Math.random() * 0xffffff ).toString( 16 );
        // this.context.fillRect( bounds.x, bounds.y, bounds.width, bounds.height );
        this.context.restore();
      }
    },
    
    getSVGString: function() {
      return '<image xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="' + this.canvas.toDataURL() + '" x="0" y="0" height="' + this.canvas.height + 'px" width="' + this.canvas.width + 'px"/>';
    },
    
    // TODO: note for DOM we can do https://developer.mozilla.org/en-US/docs/HTML/Canvas/Drawing_DOM_objects_into_a_canvas
    renderToCanvas: function( canvas, context, delayCounts ) {
      context.drawImage( this.canvas, 0, 0 );
    },
    
    markDirtyRegion: function( args ) {
      this.internalMarkDirtyBounds( args.bounds, args.transform );
    },
    
    addNodeFromTrail: function( trail ) {
      Layer.prototype.addNodeFromTrail.call( this, trail );
      
      // since the node's getBounds() are in the parent coordinate frame, we peel off the last node to get the correct (relevant) transform
      // TODO: more efficient way of getting this transform?
      this.internalMarkDirtyBounds( trail.lastNode().getBounds(), trail.slice( 0, trail.length - 1 ).getTransform() );
    },
    
    removeNodeFromTrail: function( trail ) {
      Layer.prototype.removeNodeFromTrail.call( this, trail );
      
      // since the node's getBounds() are in the parent coordinate frame, we peel off the last node to get the correct (relevant) transform
      // TODO: more efficient way of getting this transform?
      this.internalMarkDirtyBounds( trail.lastNode().getBounds(), trail.slice( 0, trail.length - 1 ).getTransform() );
    },
    
    internalMarkDirtyBounds: function( localBounds, transform ) {
      assert && assert( localBounds.isEmpty() || localBounds.isFinite(), 'Infinite (non-empty) dirty bounds passed to internalMarkDirtyBounds' );
      var globalBounds = transform.transformBounds2( localBounds );
      
      // TODO: for performance, consider more than just a single dirty bounding box
      this.dirtyBounds = this.dirtyBounds.union( globalBounds.dilated( 1 ).roundedOut() );
    },
    
    transformChange: function( args ) {
      // currently no-op, since this is taken care of by markDirtyRegion
    },
    
    getName: function() {
      return 'canvas';
    }
  } );
  
  return CanvasLayer;
} );



// Copyright 2002-2012, University of Colorado

/**
 * A DOM-based layer in the scene graph. Each layer handles dirty-region handling separately,
 * and corresponds to a single canvas / svg element / DOM element in the main container.
 * Importantly, it does not contain rendered content from a subtree of the main
 * scene graph. It only will render a contiguous block of nodes visited in a depth-first
 * manner.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/layers/DOMLayer',['require','ASSERT/assert','DOT/Bounds2','SCENERY/scenery','SCENERY/layers/Layer'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Layer = require( 'SCENERY/layers/Layer' ); // DOMLayer inherits from Layer
  
  scenery.DOMLayer = function( args ) {
    Layer.call( this, args );
    
    var width = this.$main.width();
    var height = this.$main.height();
    
    this.div = document.createElement( 'div' );
    this.$div = $( this.div );
    this.$div.width( 0 );
    this.$div.height( 0 );
    this.$div.css( 'position', 'absolute' );
    this.div.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
    this.$main.append( this.div );
    
    this.scene = args.scene;
    
    this.isDOMLayer = true;
    
    // maps trail ID => DOM element fragment
    this.idElementMap = {};
    
    // maps trail ID => Trail. trails need to be reindexed
    this.idTrailMap = {};
    
    this.trails = [];
  };
  var DOMLayer = scenery.DOMLayer;
  
  DOMLayer.prototype = _.extend( {}, Layer.prototype, {
    constructor: DOMLayer,
    
    addNodeFromTrail: function( trail ) {
      Layer.prototype.addNodeFromTrail.call( this, trail );
      trail = trail.copy();
      this.reindexTrails();
      
      var node = trail.lastNode();
      
      var element = node.getDOMElement();
      
      this.idElementMap[trail.getUniqueId()] = element;
      this.idTrailMap[trail.getUniqueId()] = trail;
      
      // walk the insertion index up the array. TODO: performance: binary search version?
      var insertionIndex;
      for ( insertionIndex = 0; insertionIndex < this.trails.length; insertionIndex++ ) {
        var otherTrail = this.trails[insertionIndex];
        otherTrail.reindex();
        var comparison = otherTrail.compare( trail );
        assert && assert( comparison !== 0, 'Trail has already been inserted into the DOMLayer' );
        if ( comparison === 1 ) { // TODO: enum values!
          break;
        }
      }
      
      if ( insertionIndex === this.div.childNodes.length ) {
        this.div.appendChild( element );
        this.trails.push( trail );
      } else {
        this.div.insertBefore( this.getElementFromTrail( this.trails[insertionIndex] ) );
        this.trails.splice( insertionIndex, 0, trail );
      }
      node.updateCSSTransform( trail.getTransform() );
    },
    
    removeNodeFromTrail: function( trail ) {
      Layer.prototype.removeNodeFromTrail.call( this, trail );
      this.reindexTrails();
      
      var element = this.getElementFromTrail( trail );
      assert && assert( element, 'Trail does not exist in the DOMLayer' );
      
      delete this.idElementMap[trail.getUniqueId];
      delete this.idTrailMap[trail.getUniqueId];
      
      this.div.removeChild( element );
      
      var removalIndex = this.getIndexOfTrail( trail );
      this.trails.splice( removalIndex, 1 );
    },
    
    getElementFromTrail: function( trail ) {
      return this.idElementMap[trail.getUniqueId()];
    },
    
    reindexTrails: function() {
      _.each( this.trails, function( trail ) {
        trail.reindex();
      } );
    },
    
    getIndexOfTrail: function( trail ) {
      // find the index where our trail is at. strict equality won't work, we want to compare differently
      var i;
      for ( i = 0; i < this.trails.length; i++ ) {
        if ( this.trails[i].compare( trail ) === 0 ) {
          return i;
        }
      }
      throw new Error( 'DOMLayer.getIndexOfTrail unable to find trail: ' + trail.toString() );
    },
    
    render: function( scene, args ) {
      // nothing at all needed here, CSS transforms taken care of when dirty regions are notified
    },
    
    dispose: function() {
      Layer.prototype.dispose.call( this );
      this.$div.detach();
    },
    
    markDirtyRegion: function( args ) {
      var node = args.node;
      var trail = args.trail;
      for ( var trailId in this.idTrailMap ) {
        var subtrail = this.idTrailMap[trailId];
        subtrail.reindex();
        if ( subtrail.isExtensionOf( trail, true ) ) {
          var element = this.idElementMap[trailId];
          
          var visible = _.every( subtrail.nodes, function( node ) { return node.isVisible(); } );
          
          if ( visible ) {
            element.style.visibility = 'visible';
          } else {
            element.style.visibility = 'hidden';
          }
        }
      }
    },
    
    transformChange: function( args ) {
      var baseTrail = args.trail;
      
      // TODO: efficiency! this computes way more matrix transforms than needed
      this.startPointer.eachTrailBetween( this.endPointer, function( trail ) {
        // bail out quickly if the trails don't match
        if ( !trail.isExtensionOf( baseTrail, true ) ) {
          return;
        }
        
        var node = trail.lastNode();
        if ( node.isPainted() ) {
          node.updateCSSTransform( trail.getTransform() );
        }
      } );
    },
    
    // TODO: consider a stack-based model for transforms?
    applyTransformationMatrix: function( matrix ) {
      // nothing at all needed here
    },
    
    getContainer: function() {
      return this.div;
    },
    
    // returns next zIndex in place. allows layers to take up more than one single zIndex
    reindex: function( zIndex ) {
      this.$div.css( 'z-index', zIndex );
      this.zIndex = zIndex;
      return zIndex + 1;
    },
    
    pushClipShape: function( shape ) {
      // TODO: clipping
    },
    
    popClipShape: function() {
      // TODO: clipping
    },
    
    getSVGString: function() {
      var data = "<svg xmlns='http://www.w3.org/2000/svg' width='" + this.$main.width() + "' height='" + this.$main.height() + "'>" +
        "<foreignObject width='100%' height='100%'>" +
        $( this.div ).html() +
        "</foreignObject></svg>";
    },
    
    // TODO: note for DOM we can do https://developer.mozilla.org/en-US/docs/HTML/Canvas/Drawing_DOM_objects_into_a_canvas
    // TODO: note that http://pbakaus.github.com/domvas/ may work better, but lacks IE support
    renderToCanvas: function( canvas, context, delayCounts ) {
      // TODO: consider not silently failing?
      // var data = "<svg xmlns='http://www.w3.org/2000/svg' width='" + this.$main.width() + "' height='" + this.$main.height() + "'>" +
      //   "<foreignObject width='100%' height='100%'>" +
      //   $( this.div ).html() +
      //   "</foreignObject></svg>";
      
      // var DOMURL = window.URL || window.webkitURL || window;
      // var img = new Image();
      // var svg = new Blob( [ data ] , { type: "image/svg+xml;charset=utf-8" } );
      // var url = DOMURL.createObjectURL( svg );
      // delayCounts.increment();
      // img.onload = function() {
      //   context.drawImage( img, 0, 0 );
      //   // TODO: this loading is delayed!!! ... figure out a solution to potentially delay?
      //   DOMURL.revokeObjectURL( url );
      //   delayCounts.decrement();
      // };
      // img.src = url;
    },
    
    getName: function() {
      return 'dom';
    }
  } );
  
  return DOMLayer;
} );



// Copyright 2002-2012, University of Colorado

/**
 * A conceptual boundary between layers, where it is optional to have information about a previous or next layer.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/layers/LayerBoundary',['require','ASSERT/assert','SCENERY/scenery'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.LayerBoundary = function() {
    // layer types before and after the boundary. null indicates the lack of information (first or last layer)
    this.previousLayerType = null;
    this.nextLayerType = null;
    
    // trails to the closest nodes with isPainted() === true before and after the boundary
    this.previousPaintedTrail = null;
    this.nextPaintedTrail = null;
    
    // the TrailPointers where the previous layer was ended and the next layer begins (the trail, and enter() or exit())
    this.previousEndPointer = null;
    this.nextStartPointer = null;
  };
  var LayerBoundary = scenery.LayerBoundary;
  
  LayerBoundary.prototype = {
    constructor: LayerBoundary,
    
    hasPrevious: function() {
      return !!this.previousPaintedTrail;
    },
    
    hasNext: function() {
      return !!this.nextPaintedTrail;
    },
    
    // assumes that trail is reindexed
    equivalentPreviousTrail: function( trail ) {
      if ( this.previousPaintedTrail && trail ) {
        this.previousPaintedTrail.reindex();
        return this.previousPaintedTrail.equals( trail );
      } else {
        // check that handles null versions properly
        return this.previousPaintedTrail === trail;
      }
    },
    
    equivalentNextTrail: function( trail ) {
      if ( this.nextPaintedTrail && trail ) {
        this.nextPaintedTrail.reindex();
        return this.nextPaintedTrail.equals( trail );
      } else {
        // check that handles null versions properly
        return this.nextPaintedTrail === trail;
      }
    },
    
    toString: function() {
      return 'boundary:' +
             '\n    types:    ' +
                  ( this.previousLayerType ? this.previousLayerType.name : '' ) +
                  ' => ' +
                  ( this.nextLayerType ? this.nextLayerType.name : '' ) +
             '\n    trails:   ' +
                  ( this.previousPaintedTrail ? this.previousPaintedTrail.getUniqueId() : '' ) +
                  ' => ' +
                  ( this.nextPaintedTrail ? this.nextPaintedTrail.getUniqueId() : '' ) +
             '\n    pointers: ' +
                  ( this.previousEndPointer ? this.previousEndPointer.toString() : '' ) +
                  ' => ' +
                  ( this.nextStartPointer ? this.nextStartPointer.toString() : '' );
    }
  };
  
  return LayerBoundary;
} );

// Copyright 2002-2012, University of Colorado

/**
 * A layer state is used to construct layer information (and later, layers), and is a state machine
 * that layer strategies from each node modify. Iterating through all of the nodes in a depth-first
 * manner will modify the LayerBuilder so that layer information can be retrieved.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/layers/LayerBuilder',['require','ASSERT/assert','SCENERY/scenery','SCENERY/layers/LayerBoundary','SCENERY/util/Trail','SCENERY/util/TrailPointer'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/layers/LayerBoundary' );
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/util/TrailPointer' );
  
  /*
   * Builds layer information between trails
   *
   * previousLayerType should be null if there is no previous layer.
   */
  scenery.LayerBuilder = function( scene, previousLayerType, previousPaintedTrail, nextPaintedTrail ) {
    
    /*---------------------------------------------------------------------------*
    * Initial state
    *----------------------------------------------------------------------------*/
    
    this.layerTypeStack = [];
    this.boundaries = [];
    this.pendingBoundary = new scenery.LayerBoundary();
    this.pendingBoundary.previousLayerType = previousLayerType;
    this.pendingBoundary.previousPaintedTrail = previousPaintedTrail;
    
    /*
     * The current layer type active, and whether it has been 'used' yet. A node with isPainted() will trigger a 'used' action,
     * and if the layer hasn't been used, it will actually trigger a boundary creation. We want to collapse 'unused' layers
     * and boundaries together, so that every created layer has a node that displays something.
     */
    this.currentLayerType = previousLayerType;
    this.layerChangePending = previousPaintedTrail === null;
    
    /*---------------------------------------------------------------------------*
    * Start / End pointers
    *----------------------------------------------------------------------------*/
    
    if ( previousPaintedTrail ) {
      // Move our start pointer just past the previousPaintedTrail, since our previousLayerType is presumably for that trail's node's self.
      // Anything after that self could have been collapsed, so we need to start there.
      this.startPointer = new scenery.TrailPointer( previousPaintedTrail.copy(), true );
      this.startPointer.nestedForwards();
    } else {
      this.startPointer = new scenery.TrailPointer( new scenery.Trail( scene ), true );
    }
    
    if ( nextPaintedTrail ) {
      // include the nextPaintedTrail's 'before' in our iteration, so we can stitch properly with the next layer
      this.endPointer = new scenery.TrailPointer( nextPaintedTrail.copy(), true );
    } else {
      this.endPointer = new scenery.TrailPointer( new scenery.Trail( scene ), false );
    }
    
    this.includesEndTrail = nextPaintedTrail !== null;
    
    /*
     * LayerBoundary properties and assurances:
     *
     * previousLayerType  - initialized in constructor (in case there are no layer changes)
     *                      set in layerChange for "fresh" pending boundary
     * nextLayerType      - set and overwrites in switchToType, for collapsing layers
     *                      not set anywhere else, so we can leave it null
     * previousPaintedTrail  - initialized in constructor
     *                      updated in markPainted if there is no pending change (don't set if there is a pending change)
     * nextPaintedTrail      - set on layerChange for "stale" boundary
     *                      stays null if nextPaintedTrail === null
     * previousEndPointer - (normal boundary) set in switchToType if there is no layer change pending
     *                      set in finalization if nextPaintedTrail === null && !this.layerChangePending (previousEndPointer should be null in that case)
     * nextStartPointer   - set in switchToType (always), overwrites values so we collapse layers nicely
     */
  };
  var LayerBuilder = scenery.LayerBuilder;
  
  LayerBuilder.prototype = {
    constructor: LayerBuilder,
    
    // walks part of the state up to just before the startPointer. we want the preferred layer stack to be in place, but the rest is not important
    prepareLayerStack: function() {
      var pointer = new scenery.TrailPointer( new scenery.Trail( this.startPointer.trail.rootNode() ), true );
      
      // if the start pointer is going to execute an exit() instead of an enter() on its trail node, we need to bump up the layer stack an additional step
      var targetLength = this.startPointer.trail.length - ( this.startPointer.isBefore ? 1 : 0 );
      
      while ( pointer.trail.length <= targetLength ) {
        var node = pointer.trail.lastNode();
        if ( node.layerStrategy.hasPreferredLayerType( pointer, this ) ) {
          this.pushPreferredLayerType( node.layerStrategy.getPreferredLayerType( pointer, this ) );
        }
        pointer.trail.addDescendant( this.startPointer.trail.nodes[pointer.trail.length] );
      }
    },
    
    run: function() {
      var builder = this;
      
      // push preferred layers for ancestors of our start pointer
      this.prepareLayerStack();
      
      // console.log( '         stack: ' + _.map( builder.layerTypeStack, function( type ) { return type.name; } ).join( ', ' ) );
      
      builder.startPointer.depthFirstUntil( builder.endPointer, function( pointer ) {
        var node = pointer.trail.lastNode();
        
        if ( pointer.isBefore ) {
          // console.log( 'builder: enter ' + pointer.toString() );
          node.layerStrategy.enter( pointer, builder );
        } else {
          // console.log( 'builder: exit ' + pointer.toString() );
          node.layerStrategy.exit( pointer, builder );
        }
        // console.log( '         stack: ' + _.map( builder.layerTypeStack, function( type ) { return type.name; } ).join( ', ' ) );
      }, false ); // include the endpoints
      
      // special case handling if we are at the 'end' of the scene, so that we create another 'wrapping' boundary
      if ( !this.includesEndTrail ) {
        // console.log( 'builder: not including end trail' );
        this.pendingBoundary.previousEndPointer = builder.endPointer; // TODO: consider implications if we leave this null, to indicate that it is not ended?
        this.layerChange( null );
      }
    },
    
    // allows paintedPointer === null at the end if the main iteration's nextPaintedTrail === null (i.e. we are at the end of the scene)
    layerChange: function( paintedPointer ) {
      this.layerChangePending = false;
      
      var confirmedBoundary = this.pendingBoundary;
      
      confirmedBoundary.nextPaintedTrail = paintedPointer ? paintedPointer.trail.copy() : null;
      
      this.boundaries.push( confirmedBoundary );
      
      this.pendingBoundary = new scenery.LayerBoundary();
      this.pendingBoundary.previousLayerType = confirmedBoundary.nextLayerType;
      this.pendingBoundary.previousPaintedTrail = confirmedBoundary.nextPaintedTrail;
      // console.log( 'builder:   added boundary' );
    },
    
    /*---------------------------------------------------------------------------*
    * API for layer strategy or other interaction
    *----------------------------------------------------------------------------*/
    
    switchToType: function( pointer, layerType ) {
      this.currentLayerType = layerType;
      
      this.pendingBoundary.nextLayerType = layerType;
      this.pendingBoundary.nextStartPointer = pointer.copy();
      if ( !this.layerChangePending ) {
        this.pendingBoundary.previousEndPointer = pointer.copy();
      }
      
      this.layerChangePending = true; // we wait until the first markPainted() call to create a boundary
    },
    
    // called so that we can finalize a layer switch (instead of collapsing unneeded layers)
    markPainted: function( pointer ) {
      if ( this.layerChangePending ) {
        this.layerChange( pointer );
      } else {
        // TODO: performance-wise, don't lookup indices on this copy? make a way to create a lightweight copy?
        this.pendingBoundary.previousPaintedTrail = pointer.trail.copy();
      }
    },
    
    // can be null to indicate that there is no current layer type
    getCurrentLayerType: function() {
      return this.currentLayerType;
    },
    
    pushPreferredLayerType: function( layerType ) {
      this.layerTypeStack.push( layerType );
    },
    
    popPreferredLayerType: function() {
      this.layerTypeStack.pop();
    },
    
    getPreferredLayerType: function() {
      if ( this.layerTypeStack.length !== 0 ) {
        return this.layerTypeStack[this.layerTypeStack.length - 1];
      } else {
        return null;
      }
    },
    
    bestPreferredLayerTypeFor: function( renderers ) {
      for ( var i = this.layerTypeStack.length - 1; i >= 0; i-- ) {
        var preferredType = this.layerTypeStack[i];
        if ( _.some( renderers, function( renderer ) { return preferredType.supportsRenderer( renderer ); } ) ) {
          return preferredType;
        }
      }
      
      // none of our stored preferred layer types are able to support any of the default type options
      return null;
    }
  };
  
  return LayerBuilder;
} );

// Copyright 2002-2012, University of Colorado

/**
 * A description of layer settings and the ability to create a layer with those settings.
 * Used internally for the layer building process.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/layers/LayerType',['require','ASSERT/assert','SCENERY/scenery'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.LayerType = function( Constructor, name, renderer, args ) {
    this.Constructor = Constructor;
    this.name = name;
    this.renderer = renderer;
    this.args = args;
  };
  var LayerType = scenery.LayerType;
  
  LayerType.prototype = {
    constructor: LayerType,
    
    supportsRenderer: function( renderer ) {
      return this.renderer === renderer;
    },
    
    supportsNode: function( node ) {
      var that = this;
      return _.some( node._supportedRenderers, function( renderer ) {
        return that.supportsRenderer( renderer );
      } );
    },
    
    createLayer: function( args ) {
      var Constructor = this.Constructor;
      return new Constructor( _.extend( {}, args, this.args ) ); // allow overriding certain arguments if necessary by the LayerType
    }
  };
  
  return LayerType;
} );



// Copyright 2002-2012, University of Colorado

/**
 * A DOM-based layer in the scene graph. Each layer handles dirty-region handling separately,
 * and corresponds to a single canvas / svg element / DOM element in the main container.
 * Importantly, it does not contain rendered content from a subtree of the main
 * scene graph. It only will render a contiguous block of nodes visited in a depth-first
 * manner.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/layers/SVGLayer',['require','ASSERT/assert','DOT/Bounds2','DOT/Transform3','DOT/Matrix3','SCENERY/scenery','SCENERY/layers/Layer','SCENERY/util/Trail'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Transform3 = require( 'DOT/Transform3' );
  var Matrix3 = require( 'DOT/Matrix3' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Layer = require( 'SCENERY/layers/Layer' ); // extends Layer
  require( 'SCENERY/util/Trail' );
  
  // used namespaces
  var svgns = 'http://www.w3.org/2000/svg';
  var xlinkns = 'http://www.w3.org/1999/xlink';
  
  scenery.SVGLayer = function( args ) {
    var $main = args.$main;
    
    // main SVG element
    this.svg = document.createElementNS( svgns, 'svg' );
    
    // the SVG has a single group under it, which corresponds to the transform of the layer's base node
    // TODO: consider renaming to 'this.baseGroup'
    this.g = document.createElementNS( svgns, 'g' );
    
    // the <defs> block that we will be stuffing gradients and patterns into
    this.defs = document.createElementNS( svgns, 'defs' );
    
    var width = $main.width();
    var height = $main.height();
    
    this.svg.appendChild( this.defs );
    this.svg.appendChild( this.g );
    this.$svg = $( this.svg );
    this.svg.setAttribute( 'width', width );
    this.svg.setAttribute( 'height', height );
    this.svg.setAttribute( 'stroke-miterlimit', 10 ); // to match our Canvas brethren so we have the same default behavior
    this.$svg.css( 'position', 'absolute' );
    this.svg.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
    this.svg.style['pointer-events'] = 'none';
    $main.append( this.svg );
    
    this.scene = args.scene;
    
    this.isSVGLayer = true;
    
    // maps trail ID => SVG self fragment (that displays shapes, text, etc.)
    this.idFragmentMap = {};
    
    // maps trail ID => SVG <g> that contains that node's self and everything under it
    this.idGroupMap = {};
    
    
    Layer.call( this, args );
    
    this.baseTransformDirty = true;
    this.baseTransformChange = true;
  };
  var SVGLayer = scenery.SVGLayer;
  
  SVGLayer.prototype = _.extend( {}, Layer.prototype, {
    constructor: SVGLayer,
    
    /*
     * Notes about how state is tracked here:
     * Trails are stored on group.trail so that we can look this up when inserting new groups
     */
    addNodeFromTrail: function( trail ) {
      assert && assert( !( trail.getUniqueId() in this.idFragmentMap ), 'Already contained that trail!' );
      assert && assert( trail.isPainted(), 'Don\'t add nodes without isPainted() to SVGLayer' );
      
      Layer.prototype.addNodeFromTrail.call( this, trail );
      
      var subtrail = this.baseTrail.copy(); // grab the trail up to (and including) the base node, so we don't create superfluous groups
      var lastId = null;
      
      // walk a subtrail up from the root node all the way to the full trail, creating groups where necessary
      while ( subtrail.length <= trail.length ) {
        var id = subtrail.getUniqueId();
        if ( !( id in this.idGroupMap ) ) {
          // we need to create a new group
          var group;
          
          if ( lastId ) {
            // we have a parent group to which we need to be added
            group = document.createElementNS( svgns, 'g' );
            
            // apply the node's transform to the group
            this.applyTransform( subtrail.lastNode().getTransform(), group );
            
            // add the group to its parent
            this.insertGroupIntoParent( group, this.idGroupMap[lastId], subtrail );
          } else {
            // we are ensuring the base group
            assert && assert( subtrail.lastNode() === this.baseNode );
            
            group = this.g;
            
            // sets up the proper transform for the base
            this.initializeBase();
          }
          
          group.referenceCount = 0; // initialize a reference count, so we can know when to remove unused groups
          group.trail = subtrail.copy(); // put a reference to the trail on the group, so we can efficiently scan and see where to insert future groups
          
          this.idGroupMap[id] = group;
        }
        
        // this trail will depend on this group, so increment the reference counter
        this.idGroupMap[id].referenceCount++;
        
        // step down towards our full trail
        subtrail.addDescendant( trail.nodes[subtrail.length] );
        lastId = id;
      }
      
      // actually add the node into its own group
      var node = trail.lastNode();
      var trailId = trail.getUniqueId();
      
      var nodeGroup = this.idGroupMap[trailId];
      var svgFragment = node.createSVGFragment( this.svg, this.defs, nodeGroup );
      this.updateNode( node, svgFragment );
      this.updateNodeGroup( node, nodeGroup );
      this.idFragmentMap[trailId] = svgFragment;
      nodeGroup.appendChild( svgFragment );
    },
    
    removeNodeFromTrail: function( trail ) {
      assert && assert( trail.getUniqueId() in this.idFragmentMap, 'Did not contain that trail!' );
      
      Layer.prototype.removeNodeFromTrail.call( this, trail );
      
      // clean up the fragment and defs directly died to the node
      var trailId = trail.getUniqueId();
      var node = trail.lastNode();
      var fragment = this.idFragmentMap[trailId];
      this.idGroupMap[trailId].removeChild( fragment );
      delete this.idFragmentMap[trailId];
      if ( node.removeSVGDefs ) {
        node.removeSVGDefs( this.svg, this.defs );
      }
      
      // clean up any unneeded groups
      var subtrail = trail.copy();
      while ( subtrail.length > this.baseTrail.length ) {
        var id = subtrail.getUniqueId();
        
        var group = this.idGroupMap[id];
        group.referenceCount--;
        if ( group.referenceCount === 0 ) {
          // completely kill the group
          group.parentNode.removeChild( group );
          delete group.trail; // just in case someone held a reference
          delete this.idGroupMap[id];
        }
        
        subtrail.removeDescendant();
      }
      this.g.referenceCount--; // since we don't go down to the base group, adjust its reference count
    },
    
    // subtrail is to group, and should include parentGroup below
    insertGroupIntoParent: function( group, parentGroup, subtrail ) {
      if ( !parentGroup.childNodes.length ) {
        parentGroup.appendChild( group );
      } else {
        // if there is already a child, we need to do a scan to ensure we place our group as a child in the correct order (above/below)
        
        // scan other child groups in the parentGroup to find where we need to be (index i)
        var afterNode = null;
        var indexIndex = subtrail.length - 2; // index into the trail's indices
        var ourIndex = subtrail.indices[indexIndex];
        var i;
        for ( i = 0; i < parentGroup.childNodes.length; i++ ) {
          var child = parentGroup.childNodes[i];
          if ( child.trail ) {
            child.trail.reindex();
            var otherIndex = child.trail.indices[indexIndex];
            if ( otherIndex > ourIndex ) {
              // this other group is above us
              break;
            }
          }
        }
        
        // insert our group before parentGroup.childNodes[i] (or append if that doesn't exist)
        if ( i === parentGroup.childNodes.length ) {
          parentGroup.appendChild( group );
        } else {
          parentGroup.insertBefore( group, parentGroup.childNodes[i] );
        }
      }
    },
    
    // updates visual styles on an existing SVG fragment
    updateNode: function( node, fragment ) {
      if ( node.updateSVGFragment ) {
        node.updateSVGFragment( fragment );
      }
      if ( node.updateSVGDefs ) {
        node.updateSVGDefs( this.svg, this.defs );
      }
    },
    
    // updates necessary paint attributes on a group (not including transform)
    updateNodeGroup: function( node, group ) {
      if ( node.isVisible() ) {
        group.style.display = 'inherit';
      } else {
        group.style.display = 'none';
      }
      group.setAttribute( 'opacity', node.getOpacity() );
    },
    
    applyTransform: function( transform, group ) {
      if ( transform.isIdentity() ) {
        if ( group.hasAttribute( 'transform' ) ) {
          group.removeAttribute( 'transform' );
        }
      } else {
        group.setAttribute( 'transform', transform.getMatrix().getSVGTransform() );
      }
    },
    
    render: function( scene, args ) {
      var layer = this;
      if ( this.baseTransformDirty ) {
        // this will be run either now or at the end of flushing changes
        var includesBaseTransformChange = this.baseTransformChange;
        this.domChange( function() {
          layer.updateBaseTransform( includesBaseTransformChange );
        } );
        
        this.baseTransformDirty = false;
        this.baseTransformChange = false;
      }
      this.flushDOMChanges();
    },
    
    dispose: function() {
      Layer.prototype.dispose.call( this );
      this.$svg.detach();
    },
    
    markDirtyRegion: function( args ) {
      var node = args.node;
      var trailId = args.trail.getUniqueId();
      
      var fragment = this.idFragmentMap[trailId];
      var group = this.idGroupMap[trailId];
      
      if ( fragment ) {
        assert && assert( group );
        this.updateNode( node, fragment );
      }
      if ( group ) {
        this.updateNodeGroup( node, group );
      }
    },
    
    markBaseTransformDirty: function( changed ) {
      var baseTransformChange = this.baseTransformChange || !!changed;
      if ( this.batchDOMChanges ) {
        this.baseTransformDirty = true;
        this.baseTransformChange = baseTransformChange;
      } else {
        this.updateBaseTransform( baseTransformChange );
      }
    },
    
    initializeBase: function() {
      // we don't want to call updateBaseTransform() twice, since baseNodeInternalBoundsChange() will call it if we use CSS transform
      if ( this.cssTransform ) {
        this.baseNodeInternalBoundsChange();
      } else {
        this.markBaseTransformDirty( true );
      }
    },
    
    // called when the base node's "internal" (self or child) bounds change, but not when it is just from the base node's own transform changing
    baseNodeInternalBoundsChange: function() {
      if ( this.cssTransform ) {
        // we want to set the baseNodeTransform to a translation so that it maps the baseNode's self/children in the baseNode's local bounds to (0,0,w,h)
        var internalBounds = this.baseNode.parentToLocalBounds( this.baseNode.getBounds() );
        var padding = scenery.Layer.cssTransformPadding;
        
        // if there is nothing, or the bounds are empty for some reason, skip this!
        if ( !internalBounds.isEmpty() ) {
          this.baseNodeTransform.set( Matrix3.translation( Math.ceil( -internalBounds.minX + padding), Math.ceil( -internalBounds.minY + padding ) ) );
          var baseNodeInteralBounds = this.baseNodeTransform.transformBounds2( internalBounds );
          
          // sanity check to ensure we are within that range
          assert && assert( baseNodeInteralBounds.minX >= 0 && baseNodeInteralBounds.minY >= 0 );
          
          this.updateContainerDimensions( Math.ceil( baseNodeInteralBounds.maxX + padding ),
                                          Math.ceil( baseNodeInteralBounds.maxY + padding ) );
        }
        
        // if this gets removed, update initializeBase()
        this.markBaseTransformDirty( true );
      } else if ( this.usesPartialCSSTransforms ) {
        this.markBaseTransformDirty( true );
      }
    },
    
    updateContainerDimensions: function( width, height ) {
      var layer = this;
      this.domChange( function() {
        layer.svg.setAttribute( 'width', width );
        layer.svg.setAttribute( 'height', height );
      } );
    },
    
    updateBaseTransform: function( includesBaseTransformChange ) {
      var transform = this.baseTrail.getTransform();
      
      if ( this.cssTransform ) {
        // set the full transform!
        this.$svg.css( transform.getMatrix().timesMatrix( this.baseNodeTransform.getInverse() ).getCSSTransformStyles() );
        
        if ( includesBaseTransformChange ) {
          this.applyTransform( this.baseNodeTransform, this.g );
        }
      } else if ( this.usesPartialCSSTransforms ) {
        // calculate what our CSS transform should be
        var cssTransform = new Transform3();
        var matrix = transform.getMatrix();
        if ( this.cssTranslation ) {
          cssTransform.append( Matrix3.translation( matrix.m02(), matrix.m12() ) );
        }
        if ( this.cssRotation ) {
          cssTransform.append( Matrix3.rotation2( matrix.getRotation() ) );
        }
        if ( this.cssScale ) {
          var scaleVector = matrix.getScaleVector();
          cssTransform.append( Matrix3.scaling( scaleVector.x, scaleVector.y ) );
        }
        
        // take the CSS transform out of what we will apply to the group
        transform.prepend( cssTransform.getInverse() );
        
        // now we need to see where our baseNode bounds are mapped to with our transform,
        // so that we can apply an extra translation and adjust dimensions as necessary
        var padding = scenery.Layer.cssTransformPadding;
        var internalBounds = this.baseNode.parentToLocalBounds( this.baseNode.getBounds() );
        var mappedBounds = transform.transformBounds2( internalBounds );
        var translation = Matrix3.translation( Math.ceil( -mappedBounds.minX + padding ), Math.ceil( -mappedBounds.minY + padding ) );
        var inverseTranslation = translation.inverted();
        this.updateContainerDimensions( Math.ceil( mappedBounds.getWidth()  + 2 * padding ),
                                        Math.ceil( mappedBounds.getHeight() + 2 * padding ) );
        
        // put the translation adjustment and its inverse in-between the two transforms
        cssTransform.append( inverseTranslation );
        transform.prepend( translation );
        
        // apply the transforms
        // TODO: checks to make sure we don't apply them in a row if one didn't change!
        this.$svg.css( cssTransform.getMatrix().getCSSTransformStyles() );
        this.applyTransform( transform, this.g );
      } else {
        this.applyTransform( transform, this.g );
      }
    },
    
    transformChange: function( args ) {
      var layer = this;
      var node = args.node;
      var trail = args.trail;
      
      if ( trail.lastNode() === this.baseNode ) {
        // our trail points to the base node. handle this case as special
        this.markBaseTransformDirty();
      } else if ( _.contains( trail.nodes, this.baseNode ) ) {
        var group = this.idGroupMap[trail.getUniqueId()];
        
        // apply the transform to the group
        this.domChange( function() {
          layer.applyTransform( node.getTransform(), group );
        } );
      } else {
        // ancestor node changed a transform. rebuild the base transform
        this.markBaseTransformDirty();
      }
    },
    
    // TODO: consider a stack-based model for transforms?
    applyTransformationMatrix: function( matrix ) {
      // nothing at all needed here
    },
    
    getContainer: function() {
      return this.svg;
    },
    
    // returns next zIndex in place. allows layers to take up more than one single zIndex
    reindex: function( zIndex ) {
      this.$svg.css( 'z-index', zIndex );
      this.zIndex = zIndex;
      return zIndex + 1;
    },
    
    pushClipShape: function( shape ) {
      // TODO: clipping
    },
    
    popClipShape: function() {
      // TODO: clipping
    },
    
    getSVGString: function() {
      // TODO: jQuery seems to be stripping namespaces, so figure that one out?
      return $( '<div>' ).append( this.$svg.clone() ).html();
      
      // also note:
      // var doc = document.implementation.createHTMLDocument("");
      // doc.write(html);
       
      // // You must manually set the xmlns if you intend to immediately serialize the HTML
      // // document to a string as opposed to appending it to a <foreignObject> in the DOM
      // doc.documentElement.setAttribute("xmlns", doc.documentElement.namespaceURI);
       
      // // Get well-formed markup
      // html = (new XMLSerializer).serializeToString(doc);
    },
    
    // TODO: note for DOM we can do https://developer.mozilla.org/en-US/docs/HTML/Canvas/Drawing_DOM_objects_into_a_canvas
    renderToCanvas: function( canvas, context, delayCounts ) {
      // temporarily put the full transform on the containing group so the rendering is correct (CSS transforms can take this away)
      this.applyTransform( this.baseTrail.getTransform(), this.g );
      
      if ( window.canvg ) {
        delayCounts.increment();
        
        // TODO: if we are using CSS3 transforms, run that here
        canvg( canvas, this.getSVGString(), {
          ignoreMouse: true,
          ignoreAnimation: true,
          ignoreDimensions: true,
          ignoreClear: true,
          renderCallback: function() {
            delayCounts.decrement();
          }
        } );
      } else {
        // will not work on Internet Explorer 9/10
        
        // TODO: very much not convinced that this is better than setting src of image
        var DOMURL = window.URL || window.webkitURL || window;
        var img = new Image();
        var raw = this.getSVGString();
        console.log( raw );
        var svg = new Blob( [ raw ] , { type: "image/svg+xml;charset=utf-8" } );
        var url = DOMURL.createObjectURL( svg );
        delayCounts.increment();
        img.onload = function() {
          context.drawImage( img, 0, 0 );
          // TODO: this loading is delayed!!! ... figure out a solution to potentially delay?
          DOMURL.revokeObjectURL( url );
          delayCounts.decrement();
        };
        img.src = url;
        
        throw new Error( 'this implementation hits Chrome bugs, won\'t work on IE9/10, etc. deprecated' );
      }
      
      // revert the transform damage that we did to our base group
      this.updateBaseTransform();
    },
    
    getName: function() {
      return 'svg';
    }
  } );
  
  return SVGLayer;
} );



// Copyright 2002-2012, University of Colorado

/**
 * An enumeration of different back-end technologies used for rendering. It also essentially
 * represents the API that nodes need to implement to be used with this specified back-end.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/layers/Renderer',['require','ASSERT/assert','SCENERY/scenery','SCENERY/layers/LayerType','SCENERY/layers/CanvasLayer','SCENERY/layers/DOMLayer','SCENERY/layers/SVGLayer'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  require( 'SCENERY/layers/LayerType' );
  require( 'SCENERY/layers/CanvasLayer' );
  require( 'SCENERY/layers/DOMLayer' );
  require( 'SCENERY/layers/SVGLayer' );
  
  // cached defaults
  var defaults = {};
  
  scenery.Renderer = function( layerConstructor, name, defaultOptions ) {
    this.layerConstructor = layerConstructor;
    this.name = name;
    this.defaultOptions = defaultOptions;
    
    this.defaultLayerType = this.createLayerType( {} ); // default options are handled in createLayerType
  };
  var Renderer = scenery.Renderer;
  
  Renderer.prototype = {
    constructor: Renderer,
    
    createLayerType: function( rendererOptions ) {
      return new scenery.LayerType( this.layerConstructor, this.name, this, _.extend( {}, this.defaultOptions, rendererOptions ) );
    }
  };
  
  Renderer.Canvas = new Renderer( scenery.CanvasLayer, 'canvas', {} );
  Renderer.DOM = new Renderer( scenery.DOMLayer, 'dom', {} );
  Renderer.SVG = new Renderer( scenery.SVGLayer, 'svg', {} );
  
  // add shortcuts for the default layer types
  scenery.CanvasDefaultLayerType = Renderer.Canvas.defaultLayerType;
  scenery.DOMDefaultLayerType    = Renderer.DOM.defaultLayerType;
  scenery.SVGDefaultLayerType    = Renderer.SVG.defaultLayerType;
  
  // and shortcuts so we can index in with shorthands like 'svg', 'dom', etc.
  Renderer.canvas = Renderer.Canvas;
  Renderer.dom = Renderer.DOM;
  Renderer.svg = Renderer.SVG;
  Renderer.webgl = Renderer.WebGL;
  
  return Renderer;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Like Underscore's _.extend, but with hardcoded support for ES5 getters/setters.
 *
 * See https://github.com/documentcloud/underscore/pull/986.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('PHET_CORE/extend',['require'], function( require ) {
  
  
  return function extend( obj ) {
    _.each( Array.prototype.slice.call( arguments, 1 ), function( source ) {
      if ( source ) {
        for ( var prop in source ) {
          Object.defineProperty( obj, prop, Object.getOwnPropertyDescriptor( source, prop ) );
        }
      }
    });
    return obj;
  };
} );

// Copyright 2013, University of Colorado

/**
 * Experimental prototype inheritance
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */
define('PHET_CORE/inherit',['require','PHET_CORE/extend'], function( require ) {
  
  
  var extend = require( 'PHET_CORE/extend' );
  
  /**
   * Experimental inheritance prototype, similar to Inheritance.inheritPrototype, but maintains
   * supertype.prototype.constructor while properly copying ES5 getters and setters.
   *
   * TODO: find problems with this! It's effectively what is being used by Scenery
   *
   * Usage:
   * function A() { scenery.Node.call( this ); };
   * inherit( A, scenery.Node, {
   *   customBehavior: function() { ... },
   *   isAnA: true
   * } );
   * new A().isAnA // true
   * new scenery.Node().isAnA // undefined
   * new A().constructor.name // 'A'
   *
   * @param subtype             Constructor for the subtype. Generally should contain supertype.call( this, ... )
   * @param supertype           Constructor for the supertype.
   * @param prototypeProperties [optional] object containing properties that will be set on the prototype.
   */
  function inherit( subtype, supertype, prototypeProperties ) {
    function F() {}
    F.prototype = supertype.prototype; // so new F().__proto__ === supertype.prototype
    
    subtype.prototype = extend( // extend will combine the properties and constructor into the new F copy
      new F(),                  // so new F().__proto__ === supertype.prototype, and the prototype chain is set up nicely
      { constructor: subtype }, // overrides the constructor properly
      prototypeProperties       // [optional] additional properties for the prototype, as an object.
    );
  }

  return inherit;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Mix-in for nodes that support a standard fill.
 *
 * TODO: pattern and gradient handling
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/nodes/Fillable',['require','ASSERT/assert','SCENERY/scenery'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Fillable = function( type ) {
    var proto = type.prototype;
    
    // this should be called in the constructor to initialize
    proto.initializeFillable = function() {
      this._fill = null;
    };
    
    proto.hasFill = function() {
      return this._fill !== null;
    };
    
    proto.getFill = function() {
      return this._fill;
    };
    
    proto.setFill = function( fill ) {
      if ( this.getFill() !== fill ) {
        this._fill = fill;
        this.invalidatePaint();
        
        this.invalidateFill();
      }
      return this;
    };
    
    proto.beforeCanvasFill = function( wrapper ) {
      wrapper.setFillStyle( this._fill );
      if ( this._fill.transformMatrix ) {
        wrapper.context.save();
        this._fill.transformMatrix.canvasAppendTransform( wrapper.context );
      }
    };
    
    proto.afterCanvasFill = function( wrapper ) {
      if ( this._fill.transformMatrix ) {
        wrapper.context.restore();
      }
    };
    
    proto.getSVGFillStyle = function() {
      // if the fill has an SVG definition, use that with a URL reference to it
      return 'fill: ' + ( this._fill ? ( this._fill.getSVGDefinition ? 'url(#fill' + this.getId() + ')' : this._fill ) : 'none' ) + ';';
    };
    
    proto.addSVGFillDef = function( svg, defs ) {
      var fill = this.getFill();
      var fillId = 'fill' + this.getId();
      
      // add new definitions if necessary
      if ( fill && fill.getSVGDefinition ) {
        defs.appendChild( fill.getSVGDefinition( fillId ) );
      }
    };
    
    proto.removeSVGFillDef = function( svg, defs ) {
      var fillId = 'fill' + this.getId();
      
      // wipe away any old definition
      var oldFillDef = svg.getElementById( fillId );
      if ( oldFillDef ) {
        defs.removeChild( oldFillDef );
      }
    };
    
    proto.appendFillablePropString = function( spaces, result ) {
      if ( this._fill ) {
        if ( result ) {
          result += ',\n';
        }
        if ( typeof this._fill === 'string' ) {
          result += spaces + 'fill: \'' + this._fill + '\'';
        } else {
          result += spaces + 'fill: ' + this._fill.toString();
        }
      }
      
      return result;
    };
    
    // on mutation, set the fill parameter first
    proto._mutatorKeys = [ 'fill' ].concat( proto._mutatorKeys );
    
    Object.defineProperty( proto, 'fill', { set: proto.setFill, get: proto.getFill } );
    
    if ( !proto.invalidateFill ) {
      proto.invalidateFill = function() {
        // override if fill handling is necessary (TODO: mixins!)
      };
    }
  };
  var Fillable = scenery.Fillable;
  
  return Fillable;
} );



// Copyright 2002-2012, University of Colorado

/**
 * Mix-in for nodes that support a standard stroke.
 *
 * TODO: miterLimit handling
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/nodes/Strokable',['require','ASSERT/assert','SCENERY/scenery','KITE/util/LineStyles'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  var LineStyles = require( 'KITE/util/LineStyles' );
  
  scenery.Strokable = function( type ) {
    var proto = type.prototype;
    
    // this should be called in the constructor to initialize
    proto.initializeStrokable = function() {
      this._stroke = null;
      this._lineDrawingStyles = new LineStyles();
    };
    
    proto.hasStroke = function() {
      return this._stroke !== null;
    };
    
    // TODO: setting these properties looks like a good candidate for refactoring to lessen file size
    proto.getLineWidth = function() {
      return this._lineDrawingStyles.lineWidth;
    };
    
    proto.setLineWidth = function( lineWidth ) {
      if ( this.getLineWidth() !== lineWidth ) {
        this.markOldSelfPaint(); // since the previous line width may have been wider
        
        this._lineDrawingStyles.lineWidth = lineWidth;
        
        this.invalidateStroke();
      }
      return this;
    };
    
    proto.getLineCap = function() {
      return this._lineDrawingStyles.lineCap;
    };
    
    proto.setLineCap = function( lineCap ) {
      if ( this._lineDrawingStyles.lineCap !== lineCap ) {
        this.markOldSelfPaint();
        
        this._lineDrawingStyles.lineCap = lineCap;
        
        this.invalidateStroke();
      }
      return this;
    };
    
    proto.getLineJoin = function() {
      return this._lineDrawingStyles.lineJoin;
    };
    
    proto.setLineJoin = function( lineJoin ) {
      if ( this._lineDrawingStyles.lineJoin !== lineJoin ) {
        this.markOldSelfPaint();
        
        this._lineDrawingStyles.lineJoin = lineJoin;
        
        this.invalidateStroke();
      }
      return this;
    };
    
    proto.getLineDash = function() {
      return this._lineDrawingStyles.lineDash;
    };
    
    proto.setLineDash = function( lineDash ) {
      if ( this._lineDrawingStyles.lineDash !== lineDash ) {
        this.markOldSelfPaint();
        
        this._lineDrawingStyles.lineDash = lineDash;
        
        this.invalidateStroke();
      }
      return this;
    };
    
    proto.getLineDashOffset = function() {
      return this._lineDrawingStyles.lineDashOffset;
    };
    
    proto.setLineDashOffset = function( lineDashOffset ) {
      if ( this._lineDrawingStyles.lineDashOffset !== lineDashOffset ) {
        this.markOldSelfPaint();
        
        this._lineDrawingStyles.lineDashOffset = lineDashOffset;
        
        this.invalidateStroke();
      }
      return this;
    };
    
    proto.setLineStyles = function( lineStyles ) {
      // TODO: since we have been using lineStyles as mutable for now, lack of change check is good here?
      this.markOldSelfPaint();
      
      this._lineDrawingStyles = lineStyles;
      this.invalidateStroke();
      return this;
    };
    
    proto.getLineStyles = function() {
      return this._lineDrawingStyles;
    };
    
    proto.getStroke = function() {
      return this._stroke;
    };
    
    proto.setStroke = function( stroke ) {
      if ( this.getStroke() !== stroke ) {
        // since this can actually change the bounds, we need to handle a few things differently than the fill
        this.markOldSelfPaint();
        
        this._stroke = stroke;
        this.invalidateStroke();
      }
      return this;
    };
    
    proto.beforeCanvasStroke = function( wrapper ) {
      // TODO: is there a better way of not calling so many things on each stroke?
      wrapper.setStrokeStyle( this._stroke );
      wrapper.setLineWidth( this.getLineWidth() );
      wrapper.setLineCap( this.getLineCap() );
      wrapper.setLineJoin( this.getLineJoin() );
      wrapper.setLineDash( this.getLineDash() );
      wrapper.setLineDashOffset( this.getLineDashOffset() );
      if ( this._stroke.transformMatrix ) {
        wrapper.context.save();
        this._stroke.transformMatrix.canvasAppendTransform( wrapper.context );
      }
    };
    
    proto.afterCanvasStroke = function( wrapper ) {
      if ( this._stroke.transformMatrix ) {
        wrapper.context.restore();
      }
    };
    
    proto.getSVGStrokeStyle = function() {
      // if the style has an SVG definition, use that with a URL reference to it
      var style = 'stroke: ' + ( this._stroke ? ( this._stroke.getSVGDefinition ? 'url(#stroke' + this.getId() + ')' : this._stroke ) : 'none' ) + ';';
      if ( this._stroke ) {
        // TODO: don't include unnecessary directives?
        style += 'stroke-width: ' + this.getLineWidth() + ';';
        style += 'stroke-linecap: ' + this.getLineCap() + ';';
        style += 'stroke-linejoin: ' + this.getLineJoin() + ';';
        if ( this.getLineDash() ) {
          style += 'stroke-dasharray: ' + this.getLineDash().join( ',' ) + ';';
          style += 'stroke-dashoffset: ' + this.getLineDashOffset() + ';';
        }
      }
      return style;
    };
    
    proto.addSVGStrokeDef = function( svg, defs ) {
      var stroke = this.getStroke();
      var strokeId = 'stroke' + this.getId();
      
      // add new definitions if necessary
      if ( stroke && stroke.getSVGDefinition ) {
        defs.appendChild( stroke.getSVGDefinition( strokeId ) );
      }
    };
    
    proto.removeSVGStrokeDef = function( svg, defs ) {
      var strokeId = 'stroke' + this.getId();
      
      // wipe away any old definition
      var oldStrokeDef = svg.getElementById( strokeId );
      if ( oldStrokeDef ) {
        defs.removeChild( oldStrokeDef );
      }
    };
    
    proto.appendStrokablePropString = function( spaces, result ) {
      var self = this;
      
      function addProp( key, value, nowrap ) {
        if ( result ) {
          result += ',\n';
        }
        if ( !nowrap && typeof value === 'string' ) {
          result += spaces + key + ': \'' + value + '\'';
        } else {
          result += spaces + key + ': ' + value;
        }
      }
      
      if ( this._stroke ) {
        var defaultStyles = new LineStyles();
        if ( typeof this._stroke === 'string' ) {
          addProp( 'stroke', this._stroke );
        } else {
          addProp( 'stroke', this._stroke.toString(), true );
        }
        
        _.each( [ 'lineWidth', 'lineCap', 'lineJoin', 'lineDashOffset' ], function( prop ) {
          if ( self[prop] !== defaultStyles[prop] ) {
            addProp( prop, self[prop] );
          }
        } );
        
        if ( this.lineDash ) {
          addProp( 'lineDash', JSON.stringify( this.lineDash ), true );
        }
      }
      
      return result;
    };
    
    // on mutation, set the stroke parameters first since they may affect the bounds (and thus later operations)
    proto._mutatorKeys = [ 'stroke', 'lineWidth', 'lineCap', 'lineJoin', 'lineDash', 'lineDashOffset' ].concat( proto._mutatorKeys );
    
    // TODO: miterLimit support?
    Object.defineProperty( proto, 'stroke', { set: proto.setStroke, get: proto.getStroke } );
    Object.defineProperty( proto, 'lineWidth', { set: proto.setLineWidth, get: proto.getLineWidth } );
    Object.defineProperty( proto, 'lineCap', { set: proto.setLineCap, get: proto.getLineCap } );
    Object.defineProperty( proto, 'lineJoin', { set: proto.setLineJoin, get: proto.getLineJoin } );
    Object.defineProperty( proto, 'lineDash', { set: proto.setLineDash, get: proto.getLineDash } );
    Object.defineProperty( proto, 'lineDashOffset', { set: proto.setLineDashOffset, get: proto.getLineDashOffset } );
    
    if ( !proto.invalidateStroke ) {
      proto.invalidateStroke = function() {
        // override if stroke handling is necessary (TODO: mixins!)
      };
    }
  };
  var Strokable = scenery.Strokable;
  
  return Strokable;
} );



// Copyright 2002-2012, University of Colorado

/**
 * A Path draws a Shape with a specific type of fill and stroke.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/nodes/Path',['require','ASSERT/assert','PHET_CORE/inherit','SCENERY/scenery','SCENERY/nodes/Node','SCENERY/layers/Renderer','SCENERY/nodes/Fillable','SCENERY/nodes/Strokable','SCENERY/util/Util'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' );
  var Renderer = require( 'SCENERY/layers/Renderer' );
  var fillable = require( 'SCENERY/nodes/Fillable' );
  var strokable = require( 'SCENERY/nodes/Strokable' );
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate;
  
  scenery.Path = function Path( options ) {
    // TODO: consider directly passing in a shape object (or at least handling that case)
    this._shape = null;
    
    // ensure we have a parameter object
    options = options || {};
    
    this.initializeStrokable();
    
    Node.call( this, options );
  };
  var Path = scenery.Path;
  
  inherit( Path, Node, {
    // sets the shape drawn, or null to remove the shape
    setShape: function( shape ) {
      if ( this._shape !== shape ) {
        this._shape = shape;
        this.invalidateShape();
      }
      return this;
    },
    
    getShape: function() {
      return this._shape;
    },
    
    invalidateShape: function() {
      this.markOldSelfPaint();
      
      if ( this.hasShape() ) {
        this.invalidateSelf( this._shape.computeBounds( this._stroke ? this._lineDrawingStyles : null ) );
        this.invalidatePaint();
      }
    },
    
    // hook stroke mixin changes to invalidation
    invalidateStroke: function() {
      this.invalidateShape();
    },
    
    hasShape: function() {
      return this._shape !== null;
    },
    
    paintCanvas: function( wrapper ) {
      var context = wrapper.context;
      
      if ( this.hasShape() ) {
        // TODO: fill/stroke delay optimizations?
        context.beginPath();
        this._shape.writeToContext( context );

        if ( this._fill ) {
          this.beforeCanvasFill( wrapper ); // defined in Fillable
          context.fill();
          this.afterCanvasFill( wrapper ); // defined in Fillable
        }
        if ( this._stroke ) {
          this.beforeCanvasStroke( wrapper ); // defined in Strokable
          context.stroke();
          this.afterCanvasStroke( wrapper ); // defined in Strokable
        }
      }
    },
    
    paintWebGL: function( state ) {
      throw new Error( 'Path.prototype.paintWebGL unimplemented' );
    },
    
    // svg element, the <defs> block, and the associated group for this node's transform
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( 'http://www.w3.org/2000/svg', 'path' );
    },
    
    updateSVGFragment: function( path ) {
      if ( this.hasShape() ) {
        path.setAttribute( 'd', this._shape.getSVGPath() );
      } else if ( path.hasAttribute( 'd' ) ) {
        path.removeAttribute( 'd' );
      }
      
      path.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
    },
    
    // support patterns, gradients, and anything else we need to put in the <defs> block
    updateSVGDefs: function( svg, defs ) {
      // remove old definitions if they exist
      this.removeSVGDefs( svg, defs );
      
      // add new ones if applicable
      this.addSVGFillDef( svg, defs );
      this.addSVGStrokeDef( svg, defs );
    },
    
    // cleans up references created with udpateSVGDefs()
    removeSVGDefs: function( svg, defs ) {
      this.removeSVGFillDef( svg, defs );
      this.removeSVGStrokeDef( svg, defs );
    },
    
    isPainted: function() {
      return true;
    },
    
    // override for computation of whether a point is inside the self content
    // point is considered to be in the local coordinate frame
    containsPointSelf: function( point ) {
      if ( !this.hasShape() ) {
        return false;
      }
      
      var result = this._shape.containsPoint( point );
      
      // also include the stroked region in the hit area if applicable
      if ( !result && this._includeStrokeInHitRegion && this.hasStroke() ) {
        result = this._shape.getStrokedShape( this._lineDrawingStyles ).containsPoint( point );
      }
      return result;
    },
    
    // whether this node's self intersects the specified bounds, in the local coordinate frame
    intersectsBoundsSelf: function( bounds ) {
      // TODO: should a shape's stroke be included?
      return this.hasShape() ? this._shape.intersectsBounds( bounds ) : false;
    },
    
    set shape( value ) { this.setShape( value ); },
    get shape() { return this.getShape(); },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Path( {' + propLines + '} )';
    },
    
    getPropString: function( spaces ) {
      var result = Node.prototype.getPropString.call( this, spaces );
      result = this.appendFillablePropString( spaces, result );
      result = this.appendStrokablePropString( spaces, result );
      if ( this._shape ) {
        if ( result ) {
          result += ',\n';
        }
        result += spaces + 'shape: ' + this._shape.toString();
      }
      return result;
    }
  } );
  
  Path.prototype._mutatorKeys = [ 'shape' ].concat( Node.prototype._mutatorKeys );
  
  Path.prototype._supportedRenderers = [ Renderer.Canvas, Renderer.SVG ];
  
  // mix in fill/stroke handling code. for now, this is done after 'shape' is added to the mutatorKeys so that stroke parameters
  // get set first
  fillable( Path );
  strokable( Path );
  
  return Path;
} );



// Copyright 2002-2012, University of Colorado

/**
 * A circular node that inherits Path, and allows for optimized drawing,
 * and improved parameter handling.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/nodes/Circle',['require','ASSERT/assert','PHET_CORE/inherit','SCENERY/scenery','SCENERY/nodes/Path','KITE/Shape'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  
  scenery.Circle = function Circle( radius, options ) {
    if ( typeof x === 'object' ) {
      // allow new Circle( { circleRadius: ... } )
      // the mutators will call invalidateCircle() and properly set the shape
      options = radius;
    } else {
      this._circleRadius = radius;
      
      // ensure we have a parameter object
      options = options || {};
      
      // fallback for non-canvas or non-svg rendering, and for proper bounds computation
      options.shape = Shape.circle( 0, 0, radius );
    }
    
    Path.call( this, options );
  };
  var Circle = scenery.Circle;
  
  inherit( Circle, Path, {
    invalidateCircle: function() {
      // setShape should invalidate the path and ensure a redraw
      this.setShape( Shape.circle( this._circleX, this._circleY, this._circleRadius ) );
    },
    
    // create a circle instead of a path, hopefully it is faster in implementations
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( 'http://www.w3.org/2000/svg', 'circle' );
    },
    
    // optimized for the circle element instead of path
    updateSVGFragment: function( circle ) {
      circle.setAttribute( 'r', this._circleRadius );
      
      circle.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
    },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Circle( ' + this._circleRadius + ', {' + propLines + '} )';
    }
  } );
  
  // TODO: refactor our this common type of code for Path subtypes
  function addCircleProp( capitalizedShort ) {
    var getName = 'getCircle' + capitalizedShort;
    var setName = 'setCircle' + capitalizedShort;
    var privateName = '_circle' + capitalizedShort;
    
    Circle.prototype[getName] = function() {
      return this[privateName];
    };
    
    Circle.prototype[setName] = function( value ) {
      this[privateName] = value;
      this.invalidateCircle();
      return this;
    };
    
    Object.defineProperty( Circle.prototype, 'circle' + capitalizedShort, {
      set: Circle.prototype[setName],
      get: Circle.prototype[getName]
    } );
  }
  
  addCircleProp( 'X' );
  addCircleProp( 'Y' );
  addCircleProp( 'Radius' );
  
  // not adding mutators for now
  Circle.prototype._mutatorKeys = [ 'circleX', 'circleY', 'circleRadius' ].concat( Path.prototype._mutatorKeys );
  
  return Circle;
} );

// Copyright 2002-2012, University of Colorado

/**
 * DOM nodes. Currently lightweight handling
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/nodes/DOM',['require','ASSERT/assert','PHET_CORE/inherit','DOT/Bounds2','SCENERY/scenery','SCENERY/nodes/Node','SCENERY/layers/Renderer','SCENERY/util/Util'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' ); // DOM inherits from Node
  var Renderer = require( 'SCENERY/layers/Renderer' );
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate;
  
  scenery.DOM = function DOM( element, options ) {
    options = options || {};
    
    this._interactive = false;
    
    // unwrap from jQuery if that is passed in, for consistency
    if ( element && element.jquery ) {
      element = element[0];
    }
    
    this._container = document.createElement( 'div' );
    this._$container = $( this._container );
    this._$container.css( 'position', 'absolute' );
    this._$container.css( 'left', 0 );
    this._$container.css( 'top', 0 );
    
    this.invalidateDOMLock = false;
    
    // so that the mutator will call setElement()
    options.element = element;
    
    // will set the element after initializing
    Node.call( this, options );
  };
  var DOM = scenery.DOM;
  
  inherit( DOM, Node, {
    // needs to be attached to the DOM tree for this to work
    calculateDOMBounds: function() {
      var boundingRect = this._element.getBoundingClientRect();
      return new Bounds2( 0, 0, boundingRect.width, boundingRect.height );
    },
    
    createTemporaryContainer: function() {
      var temporaryContainer = document.createElement( 'div' );
      $( temporaryContainer ).css( {
        display: 'hidden',
        padding: '0 !important',
        margin: '0 !important',
        position: 'absolute',
        left: 0,
        top: 0,
        width: 65535,
        height: 65535
      } );
      return temporaryContainer;
    },
    
    invalidateDOM: function() {
      // prevent this from being executed as a side-effect from inside one of its own calls
      if ( this.invalidateDOMLock ) {
        return;
      }
      this.invalidateDOMLock = true;
      
      // we will place ourselves in a temporary container to get our real desired bounds
      var temporaryContainer = this.createTemporaryContainer();
      
      // move to the temporary container
      this._container.removeChild( this._element );
      temporaryContainer.appendChild( this._element );
      document.body.appendChild( temporaryContainer );
      
      // bounds computation and resize our container to fit precisely
      var selfBounds = this.calculateDOMBounds();
      this.invalidateSelf( selfBounds );
      this._$container.width( selfBounds.getWidth() );
      this._$container.height( selfBounds.getHeight() );
      
      // move back to the main container
      document.body.removeChild( temporaryContainer );
      temporaryContainer.removeChild( this._element );
      this._container.appendChild( this._element );
      
      this.invalidateDOMLock = false;
    },
    
    getDOMElement: function() {
      return this._container;
    },
    
    updateCSSTransform: function( transform ) {
      this._$container.css( transform.getMatrix().getCSSTransformStyles() );
    },
    
    isPainted: function() {
      return true;
    },
    
    setElement: function( element ) {
      if ( this._element !== element ) {
        if ( this._element ) {
          this._container.removeChild( this._element );
        }
        
        this._element = element;
        this._$element = $( element );
        
        this._container.appendChild( this._element );
        
        // TODO: bounds issue, since this will probably set to empty bounds and thus a repaint may not draw over it
        this.invalidateDOM();  
      }

      return this; // allow chaining
    },
    
    getElement: function() {
      return this._element;
    },
    
    setInteractive: function( interactive ) {
      if ( this._interactive !== interactive ) {
        this._interactive = interactive;
        
        // TODO: anything needed here?
      }
    },
    
    isInteractive: function() {
      return this._interactive;
    },
    
    set element( value ) { this.setElement( value ); },
    get element() { return this.getElement(); },
    
    set interactive( value ) { this.setInteractive( value ); },
    get interactive() { return this.isInteractive(); },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.DOM( $( \'' + this._container.innerHTML.replace( /'/g, '\\\'' ) + '\' ), {' + propLines + '} )';
    },
    
    getPropString: function( spaces ) {
      var result = Node.prototype.getPropString.call( this, spaces );
      if ( this.interactive ) {
        if ( result ) {
          result += ',\n';
        }
        result += spaces + 'interactive: true';
      }
      return result;
    }
  } );
  
  DOM.prototype._mutatorKeys = [ 'element', 'interactive' ].concat( Node.prototype._mutatorKeys );
  
  DOM.prototype._supportedRenderers = [ Renderer.DOM ];
  
  return DOM;
} );



// Copyright 2002-2012, University of Colorado

/**
 * VBox arranges the child nodes vertically, and they can be centered, left or right justified.
 * Vertical spacing can be set as a constant or a function which depends on the adjacent nodes.
 * TODO: add an option (not enabled by default) to update layout when children or children bounds change
 *
 * @author Sam Reid
 */

define('SCENERY/nodes/HBox',['require','SCENERY/scenery','SCENERY/nodes/Node','SCENERY/util/Util'], function( require ) {
  
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate; // i.e. Object.create

  /**
   *
   * @param options Same as Node.constructor.options with the following additions:
   *
   * spacing: can be a number or a function.  If a number, then it will be the vertical spacing between each object.
   *              If a function, then the function will have the signature function(top,bottom){} which returns the spacing between adjacent pairs of items.
   * align:   How to line up the items horizontally.  One of 'center', 'left' or 'right'.  Defaults to 'center'.
   *
   * @constructor
   */
  scenery.HBox = function HBox( options ) {
    // ensure we have a parameter object
    this.options = options = _.extend( {
                                         // defaults
                                         spacing: function() { return 0; },
                                         align: 'center'
                                       }, options );

    if ( typeof options.spacing === 'number' ) {
      var spacingConstant = options.spacing;
      options.spacing = function() { return spacingConstant; };
    }

    Node.call( this, options );
    this.updateLayout();
  };
  var HBox = scenery.HBox;

  HBox.prototype = objectCreate( Node.prototype );

  HBox.prototype.updateLayout = function() {
    var minY = _.min( _.map( this.children, function( child ) {return child.y;} ) );
    var maxY = _.max( _.map( this.children, function( child ) {return child.y + child.height;} ) );
    var centerY = (maxY + minY) / 2;

    //Start at x=0 in the coordinate frame of this node.  Not possible to set this through the spacing option, instead just set it with the {y:number} option.
    var x = 0;
    for ( var i = 0; i < this.children.length; i++ ) {
      var child = this.children[i];
      child.x = x;

      //Set the position horizontally
      if ( this.options.align === 'top' ) {
        child.top = minY;
      }
      else if ( this.options.align === 'bottom' ) {
        child.bottom = maxY;
      }
      else {//default to center
        child.centerY = centerY;
      }

      //Move to the next vertical position.
      x += child.width + this.options.spacing( child, this.children[i + 1] );
    }
  };
  HBox.prototype.constructor = HBox;

  return HBox;
} );
// Copyright 2002-2012, University of Colorado

/**
 * Images
 *
 * TODO: setImage / getImage and the whole toolchain that uses that
 *
 * TODO: SVG support
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/nodes/Image',['require','ASSERT/assert','PHET_CORE/inherit','DOT/Bounds2','SCENERY/scenery','SCENERY/nodes/Node','SCENERY/layers/Renderer','SCENERY/util/Util'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' ); // Image inherits from Node
  var Renderer = require( 'SCENERY/layers/Renderer' ); // we need to specify the Renderer in the prototype
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate;
  
  /*
   * Canvas renderer supports the following as 'image':
   *     URL (string)             // works, but does NOT support bounds-based parameter object keys like 'left', 'centerX', etc.
   *                              // also necessary to force updateScene() after it has loaded
   *     HTMLImageElement         // works
   *     HTMLVideoElement         // not tested
   *     HTMLCanvasElement        // works, and forces the canvas renderer
   *     CanvasRenderingContext2D // not tested, but bad luck in past
   *     ImageBitmap              // good luck creating this. currently API for window.createImageBitmap not implemented
   * SVG renderer supports the following as 'image':
   *     URL (string)
   *     HTMLImageElement
   */
  scenery.Image = function Image( image, options ) {
    // allow not passing an options object
    options = options || {};
    
    // rely on the setImage call from the super constructor to do the setup
    if ( image ) {
      options.image = image;
    }
    
    var self = this;
    // allows us to invalidate our bounds whenever an image is loaded
    this.loadListener = function( event ) {
      self.invalidateImage();
      
      // don't leak memory!
      self._image.removeEventListener( 'load', self.loadListener );
    };
    
    Node.call( this, options );
  };
  var Image = scenery.Image;
  
  inherit( Image, Node, {
    invalidateImage: function() {
      this.invalidateSelf( new Bounds2( 0, 0, this.getImageWidth(), this.getImageHeight() ) );
    },
    
    getImage: function() {
      return this._image;
    },
    
    setImage: function( image ) {
      var self = this;
      
      if ( this._image !== image && ( typeof image !== 'string' || !this._image || image !== this._image.src ) ) {
        // don't leak memory by referencing old images
        if ( this._image ) {
          this._image.removeEventListener( 'load', this.loadListener );
        }
        
        if ( typeof image === 'string' ) {
          // create an image with the assumed URL
          var src = image;
          image = document.createElement( 'img' );
          image.addEventListener( 'load', this.loadListener );
          image.src = src;
        } else if ( image instanceof HTMLImageElement ) {
          // only add a listener if we probably haven't loaded yet
          if ( !image.width || !image.height ) {
            image.addEventListener( 'load', this.loadListener );
          }
        }
        
        // swap supported renderers if necessary
        if ( image instanceof HTMLCanvasElement ) {
          if ( !this.hasOwnProperty( '_supportedRenderers' ) ) {
            this._supportedRenderers = [ Renderer.Canvas ];
            this.markLayerRefreshNeeded();
          }
        } else {
          if ( this.hasOwnProperty( '_supportedRenderers' ) ) {
            delete this._supportedRenderers; // will leave prototype intact
            this.markLayerRefreshNeeded();
          }
        }
        
        this._image = image;
        this.invalidateImage(); // yes, if we aren't loaded yet this will give us 0x0 bounds
      }
      return this;
    },
    
    invalidateOnImageLoad: function( image ) {
      var self = this;
      var listener = function( event ) {
        self.invalidateImage();
        
        // don't leak memory!
        image.removeEventListener( listener );
      };
      image.addEventListener( listener );
    },
    
    getImageWidth: function() {
      return this._image.width;
    },
    
    getImageHeight: function() {
      return this._image.height;
    },
    
    getImageURL: function() {
      return this._image.src;
    },
    
    // signal that we are actually rendering something
    isPainted: function() {
      return true;
    },
    
    /*---------------------------------------------------------------------------*
    * Canvas support
    *----------------------------------------------------------------------------*/
    
    // TODO: add SVG / DOM support
    paintCanvas: function( wrapper ) {
      wrapper.context.drawImage( this._image, 0, 0 );
    },
    
    /*---------------------------------------------------------------------------*
    * WebGL support
    *----------------------------------------------------------------------------*/
    
    paintWebGL: function( state ) {
      throw new Error( 'paintWebGL:nimplemented' );
    },
    
    /*---------------------------------------------------------------------------*
    * SVG support
    *----------------------------------------------------------------------------*/
    
    createSVGFragment: function( svg, defs, group ) {
      var element = document.createElementNS( 'http://www.w3.org/2000/svg', 'image' );
      element.setAttribute( 'x', 0 );
      element.setAttribute( 'y', 0 );
      return element;
    },
    
    updateSVGFragment: function( element ) {
      // like <image xlink:href='http://phet.colorado.edu/images/phet-logo-yellow.png' x='0' y='0' height='127px' width='242px'/>
      var xlinkns = 'http://www.w3.org/1999/xlink';
      
      element.setAttribute( 'width', this.getImageWidth() + 'px' );
      element.setAttribute( 'height', this.getImageHeight() + 'px' );
      element.setAttributeNS( xlinkns, 'xlink:href', this.getImageURL() );
    },
    
    /*---------------------------------------------------------------------------*
    * DOM support
    *----------------------------------------------------------------------------*/
    
    getDOMElement: function() {
      this._image.style.display = 'block';
      this._image.style.position = 'absolute';
      return this._image;
    },
    
    updateCSSTransform: function( transform ) {
      $( this._image ).css( transform.getMatrix().getCSSTransformStyles() );
    },
    
    set image( value ) { this.setImage( value ); },
    get image() { return this.getImage(); },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Image( \'' + this._image.src.replace( /'/g, '\\\'' ) + '\', {' + propLines + '} )';
    }
  } );
  
  Image.prototype._mutatorKeys = [ 'image' ].concat( Node.prototype._mutatorKeys );
  
  Image.prototype._supportedRenderers = [ Renderer.Canvas, Renderer.SVG, Renderer.DOM ];
  
  // utility for others
  Image.createSVGImage = function( url, width, height ) {
    var xlinkns = 'http://www.w3.org/1999/xlink';
    var svgns = 'http://www.w3.org/2000/svg';
    
    var element = document.createElementNS( svgns, 'image' );
    element.setAttribute( 'x', 0 );
    element.setAttribute( 'y', 0 );
    element.setAttribute( 'width', width + 'px' );
    element.setAttribute( 'height', height + 'px' );
    element.setAttributeNS( xlinkns, 'xlink:href', url );
    
    return element;
  };
  
  return Image;
} );



// Copyright 2002-2012, University of Colorado

/**
 * A rectangular node that inherits Path, and allows for optimized drawing,
 * and improved rectangle handling.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/nodes/Rectangle',['require','ASSERT/assert','PHET_CORE/inherit','SCENERY/scenery','SCENERY/nodes/Path','KITE/Shape'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  
  scenery.Rectangle = function Rectangle( x, y, width, height, arcWidth, arcHeight, options ) {
    if ( typeof x === 'object' ) {
      // allow new Rectangle( { rectX: x, rectY: y, rectWidth: width, rectHeight: height, ... } )
      // the mutators will call invalidateRectangle() and properly set the shape
      options = x;
    } else if ( arguments.length < 6 ) {
      // new Rectangle( x, y, width, height, [options] )
      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._rectArcWidth = 0;
      this._rectArcHeight = 0;
      
      // ensure we have a parameter object
      options = arcWidth || {};
      
      // fallback for non-canvas or non-svg rendering, and for proper bounds computation
      options.shape = this.createRectangleShape();
    } else {
      // normal case with args (including arcWidth / arcHeight)
      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._rectArcWidth = arcWidth;
      this._rectArcHeight = arcHeight;
      
      // ensure we have a parameter object
      options = options || {};
      
      // fallback for non-canvas or non-svg rendering, and for proper bounds computation
      options.shape = this.createRectangleShape();
    }
    
    Path.call( this, options );
  };
  var Rectangle = scenery.Rectangle;
  
  inherit( Rectangle, Path, {
    isRounded: function() {
      return this._rectArcWidth !== 0 && this._rectArcHeight !== 0;
    },
    
    createRectangleShape: function() {
      if ( this.isRounded() ) {
        return Shape.roundRectangle( this._rectX, this._rectY, this._rectWidth, this._rectHeight, this._rectArcWidth, this._rectArcHeight );
      } else {
        return Shape.rectangle( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
      }
    },
    
    invalidateRectangle: function() {
      // setShape should invalidate the path and ensure a redraw
      this.setShape( this.createRectangleShape() );
    },
    
    // override paintCanvas with a faster version, since fillRect and drawRect don't affect the current default path
    paintCanvas: function( wrapper ) {
      var context = wrapper.context;
      
      // use the standard version if it's a rounded rectangle, since there is no Canvas-optimized version for that
      if ( this.isRounded() ) {
        return Path.prototype.paintCanvas.call( this, wrapper );
      }
      
      // TODO: how to handle fill/stroke delay optimizations here?
      if ( this._fill ) {
        this.beforeCanvasFill( wrapper ); // defined in Fillable
        context.fillRect( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
        this.afterCanvasFill( wrapper ); // defined in Fillable
      }
      if ( this._stroke ) {
        this.beforeCanvasStroke( wrapper ); // defined in Strokable
        context.strokeRect( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
        this.afterCanvasStroke( wrapper ); // defined in Strokable
      }
    },
    
    // create a rect instead of a path, hopefully it is faster in implementations
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( 'http://www.w3.org/2000/svg', 'rect' );
    },
    
    // optimized for the rect element instead of path
    updateSVGFragment: function( rect ) {
      // see http://www.w3.org/TR/SVG/shapes.html#RectElement
      rect.setAttribute( 'x', this._rectX );
      rect.setAttribute( 'y', this._rectY );
      rect.setAttribute( 'width', this._rectWidth );
      rect.setAttribute( 'height', this._rectHeight );
      rect.setAttribute( 'rx', this._rectArcWidth );
      rect.setAttribute( 'ry', this._rectArcHeight );
      
      rect.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
    },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Rectangle( ' + this._rectX + ', ' + this._rectY + ', ' + 
                                         this._rectWidth + ', ' + this._rectHeight + ', ' +
                                         this._rectArcWidth + ', ' + this._rectArcHeight + ', {' + propLines + '} )';
    }
    
  } );
  
  function addRectProp( capitalizedShort ) {
    var getName = 'getRect' + capitalizedShort;
    var setName = 'setRect' + capitalizedShort;
    var privateName = '_rect' + capitalizedShort;
    
    Rectangle.prototype[getName] = function() {
      return this[privateName];
    };
    
    Rectangle.prototype[setName] = function( value ) {
      this[privateName] = value;
      this.invalidateRectangle();
      return this;
    };
    
    Object.defineProperty( Rectangle.prototype, 'rect' + capitalizedShort, {
      set: Rectangle.prototype[setName],
      get: Rectangle.prototype[getName]
    } );
  }
  
  addRectProp( 'X' );
  addRectProp( 'Y' );
  addRectProp( 'Width' );
  addRectProp( 'Height' );
  addRectProp( 'ArcWidth' );
  addRectProp( 'ArcHeight' );
  
  // not adding mutators for now
  Rectangle.prototype._mutatorKeys = [ 'rectX', 'rectY', 'rectWidth', 'rectHeight', 'rectArcWidth', 'rectArcHeight' ].concat( Path.prototype._mutatorKeys );
  
  return Rectangle;
} );



// Copyright 2002-2012, University of Colorado

/**
 * Font handling for text drawing
 *
 * Examples:
 * new scenery.Font().font                      // "10px sans-serif" (the default)
 * new scenery.Font( { family: 'serif' } ).font // "10px serif"
 * new scenery.Font( { weight: 'bold' } ).font  // "bold 10px sans-serif"
 * new scenery.Font( { size: 16 } ).font        // "16px sans-serif"
 * var font = new scenery.Font( {
 *   family: '"Times New Roman", serif'
 * } );
 * font.style = 'italic';
 * font.lineHeight = 10;
 * font.font;                                   // "italic 10px/10 'Times New Roman', serif"
 * font.family;                                 // "'Times New Roman', serif"
 * font.weight;                                 // 400 (the default)
 *
 * Useful specs:
 * http://www.w3.org/TR/css3-fonts/
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/util/Font',['require','ASSERT/assert','SCENERY/scenery'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  // options from http://www.w3.org/TR/css3-fonts/
  // font-family      v ---
  // font-weight      v normal | bold | bolder | lighter | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900
  // font-stretch     v normal | ultra-condensed | extra-condensed | condensed | semi-condensed | semi-expanded | expanded | extra-expanded | ultra-expanded
  // font-style       v normal | italic | oblique
  // font-size        v <absolute-size> | <relative-size> | <length> | <percentage>
  // font-size-adjust v none | auto | <number>
  // font             v [ [ <‘font-style’> || <font-variant-css21> || <‘font-weight’> || <‘font-stretch’> ]? <‘font-size’> [ / <‘line-height’> ]? <‘font-family’> ] | caption | icon | menu | message-box | small-caption | status-bar
  //                    <font-variant-css21> = [normal | small-caps]
  // font-synthesis   v none | [ weight || style ]
  
  scenery.Font = function( options ) {
    // internal string representation
    this._font = '10px sans-serif';
    
    // span for using the browser to compute font styles
    this.$span = $( document.createElement( 'span' ) );
    
    var type = typeof options;
    if ( type === 'string' ) {
      this._font = options;
    } else if ( type === 'object' ) {
      this.mutate( options );
    }
  };
  var Font = scenery.Font;
  
  Font.prototype = {
    constructor: Font,
    
    getProperty: function( property ) {
      // sanity check, in case some CSS changed somewhere
      this.$span.css( 'font', this._font );
      
      return this.$span.css( property );
    },
    setProperty: function( property, value ) {
      // sanity check, in case some CSS changed somewhere
      this.$span.css( 'font', this._font );
      
      this.$span.css( property, value );
      this._font = this.$span.css( 'font' );
      
      return this;
    },
    
    // direct access to the font string
    getFont: function() { return this._font; },
    setFont: function( value ) { this._font = value; return this; },
    
    // using the property mechanism
    getFamily: function() { return this.getProperty( 'font-family' ); },
    setFamily: function( value ) { return this.setProperty( 'font-family', value ); },
    
    getWeight: function() { return this.getProperty( 'font-weight' ); },
    setWeight: function( value ) { return this.setProperty( 'font-weight', value ); },
    
    getStretch: function() { return this.getProperty( 'font-stretch' ); },
    setStretch: function( value ) { return this.setProperty( 'font-stretch', value ); },
    
    getStyle: function() { return this.getProperty( 'font-style' ); },
    setStyle: function( value ) { return this.setProperty( 'font-style', value ); },
    
    getSize: function() { return this.getProperty( 'font-size' ); },
    setSize: function( value ) { return this.setProperty( 'font-size', value ); },
    
    // NOTE: Canvas spec forces line-height to normal
    getLineHeight: function() { return this.getProperty( 'line-height' ); },
    setLineHeight: function( value ) { return this.setProperty( 'line-height', value ); },
    
    set font( value ) { this.setFont( value ); },
    get font() { return this.getFont(); },
    
    set weight( value ) { this.setWeight( value ); },
    get weight() { return this.getWeight(); },
    
    set family( value ) { this.setFamily( value ); },
    get family() { return this.getFamily(); },
    
    set stretch( value ) { this.setStretch( value ); },
    get stretch() { return this.getStretch(); },
    
    set style( value ) { this.setStyle( value ); },
    get style() { return this.getStyle(); },
    
    set size( value ) { this.setSize( value ); },
    get size() { return this.getSize(); },
    
    set lineHeight( value ) { this.setLineHeight( value ); },
    get lineHeight() { return this.getLineHeight(); },
    
    // TODO: move this style of mutation out into more common code, if we use it again
    mutate: function( options ) {
      var font = this;
      
      _.each( this._mutatorKeys, function( key ) {
        if ( options[key] !== undefined ) {
          font[key] = options[key];
        }
      } );
    }
  };
  
  Font.prototype._mutatorKeys = [ 'font', 'weight', 'family', 'stretch', 'style', 'size', 'lineHeight' ];
  
  return Font;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Text
 *
 * TODO: newlines
 *
 * Useful specs:
 * http://www.w3.org/TR/css3-text/
 * http://www.w3.org/TR/css3-fonts/
 * http://www.w3.org/TR/SVG/text.html
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/nodes/Text',['require','ASSERT/assert','PHET_CORE/inherit','DOT/Bounds2','SCENERY/scenery','SCENERY/nodes/Node','SCENERY/layers/Renderer','SCENERY/nodes/Fillable','SCENERY/nodes/Strokable','SCENERY/util/Util','SCENERY/util/Font','SCENERY/util/Util'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' ); // inherits from Node
  var Renderer = require( 'SCENERY/layers/Renderer' );
  var fillable = require( 'SCENERY/nodes/Fillable' );
  var strokable = require( 'SCENERY/nodes/Strokable' );
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate; // i.e. Object.create
  require( 'SCENERY/util/Font' );
  require( 'SCENERY/util/Util' ); // for canvasAccurateBounds
  
  scenery.Text = function Text( text, options ) {
    this._text         = '';                 // filled in with mutator
    this._font         = new scenery.Font(); // default font, usually 10px sans-serif
    this._textAlign    = 'start';            // start, end, left, right, center
    this._textBaseline = 'alphabetic';       // top, hanging, middle, alphabetic, ideographic, bottom
    this._direction    = 'ltr';              // ltr, rtl, inherit -- consider inherit deprecated, due to how we compute text bounds in an off-screen canvas
    
    // ensure we have a parameter object
    options = options || {};
    
    // default to black filled text
    if ( options.fill === undefined ) {
      options.fill = '#000000';
    }
    
    if ( text !== undefined ) {
      // set the text parameter so that setText( text ) is effectively called in the mutator from the super call
      options.text = text;
    }
    
    this.initializeStrokable();
    
    Node.call( this, options );
  };
  var Text = scenery.Text;
  
  inherit( Text, Node, {
    setText: function( text ) {
      if ( text !== this._text ) {
        this._text = text;
        this.invalidateText();
      }
      return this;
    },
    
    getText: function() {
      return this._text;
    },
    
    invalidateText: function() {
      // TODO: faster bounds determination? getBBox()?
      // investigate http://mudcu.be/journal/2011/01/html5-typographic-metrics/
      this.invalidateSelf( this.accurateCanvasBounds() );
    },

    paintCanvas: function( wrapper ) {
      var context = wrapper.context;
      
      // extra parameters we need to set, but should avoid setting if we aren't drawing anything
      if ( this.hasFill() || this.hasStroke() ) {
        wrapper.setFont( this._font.getFont() );
        wrapper.setTextAlign( this._textAlign );
        wrapper.setTextBaseline( this._textBaseline );
        wrapper.setDirection( this._direction );
      }
      
      if ( this.hasFill() ) {
        this.beforeCanvasFill( wrapper ); // defined in Fillable
        context.fillText( this._text, 0, 0 );
        this.afterCanvasFill( wrapper ); // defined in Fillable
      }
      if ( this.hasStroke() ) {
        this.beforeCanvasStroke( wrapper ); // defined in Strokable
        context.strokeText( this._text, 0, 0 );
        this.afterCanvasStroke( wrapper ); // defined in Strokable
      }
    },
    
    paintWebGL: function( state ) {
      throw new Error( 'Text.prototype.paintWebGL unimplemented' );
    },
    
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( 'http://www.w3.org/2000/svg', 'text' );
    },
    
    updateSVGFragment: function( element ) {
      var isRTL = this._direction === 'rtl';
      
      // make the text the only child
      while ( element.hasChildNodes() ) {
        element.removeChild( element.lastChild );
      }
      element.appendChild( document.createTextNode( this._text ) );
      
      element.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
      
      switch ( this._textAlign ) {
        case 'start':
        case 'end':
          element.setAttribute( 'text-anchor', this._textAlign ); break;
        case 'left':
          element.setAttribute( 'text-anchor', isRTL ? 'end' : 'start' ); break;
        case 'right':
          element.setAttribute( 'text-anchor', !isRTL ? 'end' : 'start' ); break;
        case 'center':
          element.setAttribute( 'text-anchor', 'middle' ); break;
      }
      switch ( this._textBaseline ) {
        case 'alphabetic':
        case 'ideographic':
        case 'hanging':
        case 'middle':
          element.setAttribute( 'dominant-baseline', this._textBaseline ); break;
        default:
          throw new Error( 'impossible to get the SVG approximate bounds for textBaseline: ' + this._textBaseline );
      }
      element.setAttribute( 'direction', this._direction );
      
      // set all of the font attributes, since we can't use the combined one
      element.setAttribute( 'font-family', this._font.getFamily() );
      element.setAttribute( 'font-size', this._font.getSize() );
      element.setAttribute( 'font-style', this._font.getStyle() );
      element.setAttribute( 'font-weight', this._font.getWeight() );
      if ( this._font.getStretch() ) {
        element.setAttribute( 'font-stretch', this._font.getStretch() );
      }
    },
    
    // support patterns, gradients, and anything else we need to put in the <defs> block
    updateSVGDefs: function( svg, defs ) {
      // remove old definitions if they exist
      this.removeSVGDefs( svg, defs );
      
      // add new ones if applicable
      this.addSVGFillDef( svg, defs );
      this.addSVGStrokeDef( svg, defs );
    },
    
    // cleans up references created with udpateSVGDefs()
    removeSVGDefs: function( svg, defs ) {
      this.removeSVGFillDef( svg, defs );
      this.removeSVGStrokeDef( svg, defs );
    },
    
    /*---------------------------------------------------------------------------*
    * Bounds
    *----------------------------------------------------------------------------*/
    
    accurateCanvasBounds: function() {
      var node = this;
      var svgBounds = this.approximateSVGBounds();
      return scenery.Util.canvasAccurateBounds( function( context ) {
        context.font = node.font;
        context.textAlign = node.textAlign;
        context.textBaseline = node.textBaseline;
        context.direction = node.direction;
        context.fillText( node.text, 0, 0 );
      }, {
        precision: 0.5,
        resolution: 128,
        initialScale: 32 / Math.max( Math.abs( svgBounds.minX ), Math.abs( svgBounds.minY ), Math.abs( svgBounds.maxX ), Math.abs( svgBounds.maxY ) )
      } );
    },
    
    approximateCanvasWidth: function() {
      // TODO: consider caching a scratch 1x1 canvas for this purpose
      var context = document.createElement( 'canvas' ).getContext( '2d' );
      context.font = this.font;
      context.textAlign = this.textAlign;
      context.textBaseline = this.textBaseline;
      context.direction = this.direction;
      return context.measureText( this.text ).width;
    },
    
    approximateSVGBounds: function() {
      var isRTL = this._direction === 'rtl';
      
      var svg = document.createElementNS( 'http://www.w3.org/2000/svg', 'svg' );
      svg.setAttribute( 'width', '1024' );
      svg.setAttribute( 'height', '1024' );
      svg.setAttribute( 'style', 'display: hidden;' ); // so we don't flash it in a visible way to the user
      
      var textElement = document.createElementNS( 'http://www.w3.org/2000/svg', 'text' );
      this.updateSVGFragment( textElement );
      
      svg.appendChild( textElement );
      
      document.body.appendChild( svg );
      var rect = textElement.getBBox();
      var result = new Bounds2( rect.x, rect.y, rect.x + rect.width, rect.y + rect.height );
      document.body.removeChild( svg );
      
      return result;
    },
    
    approximateDOMBounds: function() {
      // TODO: we can also technically support 'top' using vertical-align: top and line-height: 0 with the image, but it won't usually render otherwise
      assert && assert( this._textBaseline === 'alphabetic' );
      
      var maxHeight = 1024; // technically this will fail if the font is taller than this!
      var isRTL = this.direction === 'rtl';
      
      // <div style="position: absolute; left: 0; top: 0; padding: 0 !important; margin: 0 !important;"><span id="baselineSpan" style="font-family: Verdana; font-size: 25px;">QuipTaQiy</span><div style="vertical-align: baseline; display: inline-block; width: 0; height: 500px; margin: 0 important!; padding: 0 important!;"></div></div>
      
      var div = document.createElement( 'div' );
      $( div ).css( {
        position: 'absolute',
        left: 0,
        top: 0,
        padding: '0 !important',
        margin: '0 !important',
        display: 'hidden'
      } );
      
      var span = document.createElement( 'span' );
      $( span ).css( 'font', this.getFont() );
      span.appendChild( document.createTextNode( this.text ) );
      span.setAttribute( 'direction', this._direction );
      
      var fakeImage = document.createElement( 'div' );
      $( fakeImage ).css( {
        'vertical-align': 'baseline',
        display: 'inline-block',
        width: 0,
        height: maxHeight + 'px',
        margin: '0 !important',
        padding: '0 !important'
      } );
      
      div.appendChild( span );
      div.appendChild( fakeImage );
      
      document.body.appendChild( div );
      var rect = span.getBoundingClientRect();
      var divRect = div.getBoundingClientRect();
      var result = new Bounds2( rect.left, rect.top - maxHeight, rect.right, rect.bottom - maxHeight ).shifted( -divRect.left, -divRect.top );
      document.body.removeChild( div );
      
      var width = rect.right - rect.left;
      switch ( this._textAlign ) {
        case 'start':
          result = result.shiftedX( isRTL ? -width : 0 );
          break;
        case 'end':
          result = result.shiftedX( !isRTL ? -width : 0 );
          break;
        case 'left':
          break;
        case 'right':
          result = result.shiftedX( -width );
          break;
        case 'center':
          result = result.shiftedX( -width / 2 );
          break;
      }
      
      return result;
    },
    
    /*---------------------------------------------------------------------------*
    * Self setters / getters
    *----------------------------------------------------------------------------*/
    
    setFont: function( font ) {
      // if font is a Font instance, we actually create another copy so that modification on the original will not change this font.
      // in the future we can consider adding listeners to the font to get font change notifications.
      this._font = font instanceof scenery.Font ? new scenery.Font( font.getFont() ) : new scenery.Font( font );
      this.invalidateText();
      return this;
    },
    
    // NOTE: returns mutable copy for now, consider either immutable version, defensive copy, or note about invalidateText()
    getFont: function() {
      return this._font.getFont();
    },
    
    setTextAlign: function( textAlign ) {
      this._textAlign = textAlign;
      this.invalidateText();
      return this;
    },
    
    getTextAlign: function() {
      return this._textAlign;
    },
    
    setTextBaseline: function( textBaseline ) {
      this._textBaseline = textBaseline;
      this.invalidateText();
      return this;
    },
    
    getTextBaseline: function() {
      return this._textBaseline;
    },
    
    setDirection: function( direction ) {
      this._direction = direction;
      this.invalidateText();
      return this;
    },
    
    getDirection: function() {
      return this._direction;
    },
    
    isPainted: function() {
      return true;
    },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Text( \'' + this._text.replace( /'/g, '\\\'' ) + '\', {' + propLines + '} )';
    },
    
    getPropString: function( spaces ) {
      var result = Node.prototype.getPropString.call( this, spaces );
      result = this.appendFillablePropString( spaces, result );
      result = this.appendStrokablePropString( spaces, result );
      
      // TODO: if created again, deduplicate with Node's getPropString
      function addProp( key, value, nowrap ) {
        if ( result ) {
          result += ',\n';
        }
        if ( !nowrap && typeof value === 'string' ) {
          result += spaces + key + ': \'' + value + '\'';
        } else {
          result += spaces + key + ': ' + value;
        }
      }
      
      if ( this.font !== new scenery.Font().getFont() ) {
        addProp( 'font', this.font );
      }
      
      if ( this._textAlign !== 'start' ) {
        addProp( 'textAlign', this._textAlign );
      }
      
      if ( this._textBaseline !== 'alphabetic' ) {
        addProp( 'textBaseline', this._textBaseline );
      }
      
      if ( this._direction !== 'ltr' ) {
        addProp( 'direction', this._direction );
      }
      
      return result;
    }
  } );
  
  /*---------------------------------------------------------------------------*
  * Font setters / getters
  *----------------------------------------------------------------------------*/
  
  function addFontForwarding( propertyName, fullCapitalized, shortUncapitalized ) {
    var getterName = 'get' + fullCapitalized;
    var setterName = 'set' + fullCapitalized;
    
    Text.prototype[getterName] = function() {
      // use the ES5 getter to retrieve the property. probably somewhat slow.
      return this._font[ shortUncapitalized ];
    };
    
    Text.prototype[setterName] = function( value ) {
      // use the ES5 setter. probably somewhat slow.
      this._font[ shortUncapitalized ] = value;
      this.invalidateText();
      return this;
    };
    
    Object.defineProperty( Text.prototype, propertyName, { set: Text.prototype[setterName], get: Text.prototype[getterName] } );
  }
  
  addFontForwarding( 'fontWeight', 'FontWeight', 'weight' );
  addFontForwarding( 'fontFamily', 'FontFamily', 'family' );
  addFontForwarding( 'fontStretch', 'FontStretch', 'stretch' );
  addFontForwarding( 'fontStyle', 'FontStyle', 'style' );
  addFontForwarding( 'fontSize', 'FontSize', 'size' );
  addFontForwarding( 'lineHeight', 'LineHeight', 'lineHeight' );
  
  Text.prototype._mutatorKeys = [ 'text', 'font', 'fontWeight', 'fontFamily', 'fontStretch', 'fontStyle', 'fontSize', 'lineHeight',
                                  'textAlign', 'textBaseline', 'direction' ].concat( Node.prototype._mutatorKeys );
  
  Text.prototype._supportedRenderers = [ Renderer.Canvas, Renderer.SVG ];
  
  // font-specific ES5 setters and getters are defined using addFontForwarding above
  Object.defineProperty( Text.prototype, 'font', { set: Text.prototype.setFont, get: Text.prototype.getFont } );
  Object.defineProperty( Text.prototype, 'text', { set: Text.prototype.setText, get: Text.prototype.getText } );
  Object.defineProperty( Text.prototype, 'textAlign', { set: Text.prototype.setTextAlign, get: Text.prototype.getTextAlign } );
  Object.defineProperty( Text.prototype, 'textBaseline', { set: Text.prototype.setTextBaseline, get: Text.prototype.getTextBaseline } );
  Object.defineProperty( Text.prototype, 'direction', { set: Text.prototype.setDirection, get: Text.prototype.getDirection } );
  
  // mix in support for fills and strokes
  fillable( Text );
  strokable( Text );

  return Text;
} );



// Copyright 2002-2012, University of Colorado

/**
 * VBox arranges the child nodes vertically, and they can be centered, left or right justified.
 * Vertical spacing can be set as a constant or a function which depends on the adjacent nodes.
 * TODO: add an option (not enabled by default) to update layout when children or children bounds change
 *
 * @author Sam Reid
 */

define('SCENERY/nodes/VBox',['require','SCENERY/scenery','SCENERY/nodes/Node','SCENERY/util/Util'], function( require ) {
  
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate; // i.e. Object.create

  /**
   *
   * @param options Same as Node.constructor.options with the following additions:
   *
   * spacing: can be a number or a function.  If a number, then it will be the vertical spacing between each object.
   *              If a function, then the function will have the signature function(top,bottom){} which returns the spacing between adjacent pairs of items.
   * align:   How to line up the items horizontally.  One of 'center', 'left' or 'right'.  Defaults to 'center'.
   *
   * @constructor
   */
  scenery.VBox = function VBox( options ) {
    // ensure we have a parameter object
    this.options = options = _.extend( {
      // defaults
      spacing: function() { return 0; },
      align: 'center'
    }, options );
    
    if ( typeof options.spacing === 'number' ) {
      var spacingConstant = options.spacing;
      options.spacing = function() { return spacingConstant; };
    }

    Node.call( this, options );
    this.updateLayout();
  };
  var VBox = scenery.VBox;

  VBox.prototype = objectCreate( Node.prototype );

  VBox.prototype.updateLayout = function() {
    var minX = _.min( _.map( this.children, function( child ) {return child.x;} ) );
    var maxX = _.max( _.map( this.children, function( child ) {return child.x + child.width;} ) );
    var centerX = (maxX + minX) / 2;

    //Start at y=0 in the coordinate frame of this node.  Not possible to set this through the spacing option, instead just set it with the {y:number} option.
    var y = 0;
    for ( var i = 0; i < this.children.length; i++ ) {
      var child = this.children[i];
      child.y = y;

      //Set the position horizontally
      if ( this.options.align === 'left' ) {
        child.left = minX;
      }
      else if ( this.options.align === 'right' ) {
        child.right = maxX;
      }
      else {//default to center
        child.centerX = centerX;
      }

      //Move to the next vertical position.
      y += child.height + this.options.spacing( child, this.children[i + 1] );
    }
  };
  VBox.prototype.constructor = VBox;

  return VBox;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Encapsulates common color information and transformations.
 *
 * Consider it immutable!
 *
 * See http://www.w3.org/TR/css3-color/
 *
 * TODO: consider using https://github.com/One-com/one-color internally
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/util/Color',['require','ASSERT/assert','SCENERY/scenery','DOT/Util'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var clamp = require( 'DOT/Util' ).clamp;
  
  // r,g,b integers 0-255, 'a' float 0-1
  scenery.Color = function( r, g, b, a ) {
    
    if ( typeof r === 'string' ) {
      var str = r.replace( / /g, '' ).toLowerCase();
      var success = false;
      
      // replace colors based on keywords
      var keywordMatch = Color.colorKeywords[str];
      if ( keywordMatch ) {
        str = '#' + keywordMatch;
      }
      
      // run through the available text formats
      for ( var i = 0; i < Color.formatParsers.length; i++ ) {
        var parser = Color.formatParsers[i];
        
        var matches = parser.regexp.exec( str );
        if ( matches ) {
          parser.apply( this, matches );
          success = true;
          break;
        }
      }
      
      if ( !success ) {
        throw new Error( 'scenery.Color unable to parse color string: ' + r );
      }
    } else {
      // alpha
      this.a = a === undefined ? 255 : a;

      // bitwise handling if 3 elements aren't defined
      if ( g === undefined || b === undefined ) {
        this.r = ( r >> 16 ) && 0xFF;
        this.g = ( r >> 8 ) && 0xFF;
        this.b = ( r >> 0 ) && 0xFF;
      }
      else {
        // otherwise, copy them over
        this.r = r;
        this.g = g;
        this.b = b;
      }
    }
  };
  var Color = scenery.Color;
  
  // regex utilities
  var rgbNumber = '(-?\\d{1,3}%?)'; // syntax allows negative integers and percentages
  var aNumber = '(\\d+|\\d*\\.\\d+)'; // decimal point number. technically we allow for '255', even though this will be clamped to 1.
  var rawNumber = '(\\d{1,3})'; // a 1-3 digit number
  
  // handles negative and percentage values
  function parseRGBNumber( str ) {
    var multiplier = 1;
    
    // if it's a percentage, strip it off and handle it that way
    if ( str.charAt( str.length - 1 ) === '%' ) {
      multiplier = 2.55;
      str = str.slice( 0, str.length - 1 );
    }
    
    return Math.round( parseInt( str, 10 ) * multiplier );
  }
  
  Color.formatParsers = [
    {
      // 'transparent'
      regexp: /^transparent$/,
      apply: function( color, matches ) {
        color.setRGBA( 0, 0, 0, 0 );
      }
    },{
      // short hex form, a la '#fff'
      regexp: /^#(\w{1})(\w{1})(\w{1})$/,
      apply: function( color, matches ) {
        color.setRGBA( parseInt( matches[1] + matches[1], 16 ),
                       parseInt( matches[2] + matches[2], 16 ),
                       parseInt( matches[3] + matches[3], 16 ),
                       1 );
      }
    },{
      // long hex form, a la '#ffffff'
      regexp: /^#(\w{2})(\w{2})(\w{2})$/,
      apply: function( color, matches ) {
        color.setRGBA( parseInt( matches[1], 16 ),
                       parseInt( matches[2], 16 ),
                       parseInt( matches[3], 16 ),
                       1 );
      }
    },{
      // rgb(...)
      regexp: new RegExp( '^rgb\\(' + rgbNumber + ',' + rgbNumber + ',' + rgbNumber + '\\)$' ),
      apply: function( color, matches ) {
        color.setRGBA( parseRGBNumber( matches[1] ),
                       parseRGBNumber( matches[2] ),
                       parseRGBNumber( matches[3] ),
                       1 );
      }
    },{
      // rgba(...)
      regexp: new RegExp( '^rgba\\(' + rgbNumber + ',' + rgbNumber + ',' + rgbNumber + ',' + aNumber + '\\)$' ),
      apply: function( color, matches ) {
        color.setRGBA( parseRGBNumber( matches[1] ),
                       parseRGBNumber( matches[2] ),
                       parseRGBNumber( matches[3] ),
                       parseFloat( matches[4] ) );
      }
    },{
      // hsl(...)
      regexp: new RegExp( '^hsl\\(' + rawNumber + ',' + rawNumber + '%,' + rawNumber + '%\\)$' ),
      apply: function( color, matches ) {
        color.setHSLA( parseInt( matches[1], 10 ),
                       parseInt( matches[2], 10 ),
                       parseInt( matches[3], 10 ),
                       1 );
      }
    },{
      // hsla(...)
      regexp: new RegExp( '^hsla\\(' + rawNumber + ',' + rawNumber + '%,' + rawNumber + '%,' + aNumber + '\\)$' ),
      apply: function( color, matches ) {
        color.setHSLA( parseInt( matches[1], 10 ),
                       parseInt( matches[2], 10 ),
                       parseInt( matches[3], 10 ),
                       parseFloat( matches[4] ) );
      }
    }
  ];
  
  // see http://www.w3.org/TR/css3-color/
  Color.hueToRGB = function( m1, m2, h ) {
    if ( h < 0 ) {
      h = h + 1;
    }
    if ( h > 1 ) {
      h = h - 1;
    }
    if ( h * 6 < 1 ) {
      return m1 + ( m2 - m1 ) * h * 6;
    }
    if ( h * 2 < 1 ) {
      return m2;
    }
    if ( h * 3 < 2 ) {
      return m1 + ( m2 - m1 ) * ( 2 / 3 - h ) * 6;
    }
    return m1;
  };
  
  Color.prototype = {
    constructor: Color,
    
    // RGB integral between 0-255, alpha (float) between 0-1
    setRGBA: function( red, green, blue, alpha ) {
      this.r = Math.round( clamp( red, 0, 255 ) );
      this.g = Math.round( clamp( green, 0, 255 ) );
      this.b = Math.round( clamp( blue, 0, 255 ) );
      this.a = clamp( alpha, 0, 1 );
    },
    
    // TODO: on modification, cache this.
    getCSS: function() {
      if ( this.a === 1 ) {
        return 'rgb(' + this.r + ',' + this.g + ',' + this.b + ')';
      } else {
        var alphaString = this.a === 0 || this.a === 1 ? this.a : this.a.toFixed( 20 ); // toFixed prevents scientific notation
        return 'rgba(' + this.r + ',' + this.g + ',' + this.b + ',' + alphaString + ')';
      }
    },
    
    setHSLA: function( hue, saturation, lightness, alpha ) {
      hue = ( hue % 360 ) / 360;                    // integer modulo 360
      saturation = clamp( saturation / 100, 0, 1 ); // percentage
      lightness = clamp( lightness / 100, 0, 1 );   // percentage
      
      // see http://www.w3.org/TR/css3-color/
      var m1, m2;
      if ( lightness < 0.5 ) {
        m2 = lightness * ( saturation + 1 );
      } else {
        m2 = lightness + saturation - lightness * saturation;
      }
      m1 = lightness * 2 - m2;
      
      this.r = Math.round( Color.hueToRGB( m1, m2, hue + 1/3 ) * 255 );
      this.g = Math.round( Color.hueToRGB( m1, m2, hue ) * 255 );
      this.b = Math.round( Color.hueToRGB( m1, m2, hue - 1/3 ) * 255 );
      this.a = clamp( alpha, 0, 1 );
    },
    
    equals: function( color ) {
      return this.r === color.r && this.g === color.g && this.b === color.b && this.a === color.a;
    },
    
    withAlpha: function( alpha ) {
      return new Color( this.r, this.g, this.b, alpha );
    },
    
    brighterColor: function( factor ) {
      if ( factor < 0 || factor > 1 ) {
        throw new Error( "factor must be between 0 and 1: " + factor );
      }
      var red = Math.min( 255, this.r + Math.floor( factor * ( 255 - this.r ) ) );
      var green = Math.min( 255, this.g + Math.floor( factor * ( 255 - this.g ) ) );
      var blue = Math.min( 255, this.b + Math.floor( factor * ( 255 - this.b ) ) );
      return new Color( red, green, blue, this.a );
    },
    
    darkerColor: function( factor ) {
      if ( factor < 0 || factor > 1 ) {
        throw new Error( "factor must be between 0 and 1: " + factor );
      }
      var red = Math.max( 0, this.r - Math.floor( factor * this.r ) );
      var green = Math.max( 0, this.g - Math.floor( factor * this.g ) );
      var blue = Math.max( 0, this.b - Math.floor( factor * this.b ) );
      return new Color( red, green, blue, this.a );
    }
  };
  
  Color.basicColorKeywords = {
    aqua: '00ffff',
    black: '000000',
    blue: '0000ff',
    fuchsia: 'ff00ff',
    gray: '808080',
    green: '008000',
    lime: '00ff00',
    maroon: '800000',
    navy: '000080',
    olive: '808000',
    purple: '800080',
    red: 'ff0000',
    silver: 'c0c0c0',
    teal: '008080',
    white: 'ffffff',
    yellow: 'ffff00'
  };
  
  Color.colorKeywords = {
    aliceblue: 'f0f8ff',
    antiquewhite: 'faebd7',
    aqua: '00ffff',
    aquamarine: '7fffd4',
    azure: 'f0ffff',
    beige: 'f5f5dc',
    bisque: 'ffe4c4',
    black: '000000',
    blanchedalmond: 'ffebcd',
    blue: '0000ff',
    blueviolet: '8a2be2',
    brown: 'a52a2a',
    burlywood: 'deb887',
    cadetblue: '5f9ea0',
    chartreuse: '7fff00',
    chocolate: 'd2691e',
    coral: 'ff7f50',
    cornflowerblue: '6495ed',
    cornsilk: 'fff8dc',
    crimson: 'dc143c',
    cyan: '00ffff',
    darkblue: '00008b',
    darkcyan: '008b8b',
    darkgoldenrod: 'b8860b',
    darkgray: 'a9a9a9',
    darkgreen: '006400',
    darkkhaki: 'bdb76b',
    darkmagenta: '8b008b',
    darkolivegreen: '556b2f',
    darkorange: 'ff8c00',
    darkorchid: '9932cc',
    darkred: '8b0000',
    darksalmon: 'e9967a',
    darkseagreen: '8fbc8f',
    darkslateblue: '483d8b',
    darkslategray: '2f4f4f',
    darkturquoise: '00ced1',
    darkviolet: '9400d3',
    deeppink: 'ff1493',
    deepskyblue: '00bfff',
    dimgray: '696969',
    dodgerblue: '1e90ff',
    feldspar: 'd19275',
    firebrick: 'b22222',
    floralwhite: 'fffaf0',
    forestgreen: '228b22',
    fuchsia: 'ff00ff',
    gainsboro: 'dcdcdc',
    ghostwhite: 'f8f8ff',
    gold: 'ffd700',
    goldenrod: 'daa520',
    gray: '808080',
    green: '008000',
    greenyellow: 'adff2f',
    honeydew: 'f0fff0',
    hotpink: 'ff69b4',
    indianred : 'cd5c5c',
    indigo : '4b0082',
    ivory: 'fffff0',
    khaki: 'f0e68c',
    lavender: 'e6e6fa',
    lavenderblush: 'fff0f5',
    lawngreen: '7cfc00',
    lemonchiffon: 'fffacd',
    lightblue: 'add8e6',
    lightcoral: 'f08080',
    lightcyan: 'e0ffff',
    lightgoldenrodyellow: 'fafad2',
    lightgrey: 'd3d3d3',
    lightgreen: '90ee90',
    lightpink: 'ffb6c1',
    lightsalmon: 'ffa07a',
    lightseagreen: '20b2aa',
    lightskyblue: '87cefa',
    lightslateblue: '8470ff',
    lightslategray: '778899',
    lightsteelblue: 'b0c4de',
    lightyellow: 'ffffe0',
    lime: '00ff00',
    limegreen: '32cd32',
    linen: 'faf0e6',
    magenta: 'ff00ff',
    maroon: '800000',
    mediumaquamarine: '66cdaa',
    mediumblue: '0000cd',
    mediumorchid: 'ba55d3',
    mediumpurple: '9370d8',
    mediumseagreen: '3cb371',
    mediumslateblue: '7b68ee',
    mediumspringgreen: '00fa9a',
    mediumturquoise: '48d1cc',
    mediumvioletred: 'c71585',
    midnightblue: '191970',
    mintcream: 'f5fffa',
    mistyrose: 'ffe4e1',
    moccasin: 'ffe4b5',
    navajowhite: 'ffdead',
    navy: '000080',
    oldlace: 'fdf5e6',
    olive: '808000',
    olivedrab: '6b8e23',
    orange: 'ffa500',
    orangered: 'ff4500',
    orchid: 'da70d6',
    palegoldenrod: 'eee8aa',
    palegreen: '98fb98',
    paleturquoise: 'afeeee',
    palevioletred: 'd87093',
    papayawhip: 'ffefd5',
    peachpuff: 'ffdab9',
    peru: 'cd853f',
    pink: 'ffc0cb',
    plum: 'dda0dd',
    powderblue: 'b0e0e6',
    purple: '800080',
    red: 'ff0000',
    rosybrown: 'bc8f8f',
    royalblue: '4169e1',
    saddlebrown: '8b4513',
    salmon: 'fa8072',
    sandybrown: 'f4a460',
    seagreen: '2e8b57',
    seashell: 'fff5ee',
    sienna: 'a0522d',
    silver: 'c0c0c0',
    skyblue: '87ceeb',
    slateblue: '6a5acd',
    slategray: '708090',
    snow: 'fffafa',
    springgreen: '00ff7f',
    steelblue: '4682b4',
    tan: 'd2b48c',
    teal: '008080',
    thistle: 'd8bfd8',
    tomato: 'ff6347',
    turquoise: '40e0d0',
    violet: 'ee82ee',
    violetred: 'd02090',
    wheat: 'f5deb3',
    white: 'ffffff',
    whitesmoke: 'f5f5f5',
    yellow: 'ffff00',
    yellowgreen: '9acd32'
  };
  
  // JAVA compatibility TODO: remove after porting MS
  Color.BLACK = new Color( 0, 0, 0 );
  Color.BLUE = new Color( 0, 0, 255 );
  Color.CYAN = new Color( 0, 255, 255 );
  Color.DARK_GRAY = new Color( 64, 64, 64 );
  Color.GRAY = new Color( 128, 128, 128 );
  Color.GREEN = new Color( 0, 255, 0 );
  Color.LIGHT_GRAY = new Color( 192, 192, 192 );
  Color.MAGENTA = new Color( 255, 0, 255 );
  Color.ORANGE = new Color( 255, 200, 0 );
  Color.PINK = new Color( 255, 175, 175 );
  Color.RED = new Color( 255, 0, 0 );
  Color.WHITE = new Color( 255, 255, 255 );
  Color.YELLOW = new Color( 255, 255, 0 );
  
  return Color;
} );

// Copyright 2002-2012, University of Colorado

/**
 * A linear gradient that can be passed into the 'fill' or 'stroke' parameters.
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * TODO: reduce code sharing between gradients
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/util/LinearGradient',['require','ASSERT/assert','SCENERY/scenery','DOT/Vector2'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Vector2 = require( 'DOT/Vector2' );

  // TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ]
  scenery.LinearGradient = function( x0, y0, x1, y1 ) {
    assert && assert( isFinite( x0 ) && isFinite( y0 ) && isFinite( x1 ) && isFinite( y1 ) );
    var usesVectors = y1 === undefined;
    if ( usesVectors ) {
      assert && assert( ( x0 instanceof Vector2 ) && ( y0 instanceof Vector2 ), 'If less than 4 parameters are given, the first two parameters must be Vector2' );
    }
    this.start = usesVectors ? x0 : new Vector2( x0, y0 );
    this.end = usesVectors ? y0 : new Vector2( x1, y1 );
    
    this.stops = [];
    this.lastStopRatio = 0;
    
    // TODO: make a global spot that will have a 'useless' context for these purposes?
    this.canvasGradient = document.createElement( 'canvas' ).getContext( '2d' ).createLinearGradient( x0, y0, x1, y1 );
    
    this.transformMatrix = null;
  };
  var LinearGradient = scenery.LinearGradient;
  
  LinearGradient.prototype = {
    constructor: LinearGradient,
    
    // TODO: add color support here, instead of string?
    addColorStop: function( ratio, color ) {
      if ( this.lastStopRatio > ratio ) {
        // fail out, since browser quirks go crazy for this case
        throw new Error( 'Color stops not specified in the order of increasing ratios' );
      } else {
        this.lastStopRatio = ratio;
      }
      
      this.stops.push( { ratio: ratio, color: color } );
      this.canvasGradient.addColorStop( ratio, color );
      return this;
    },
    
    setTransformMatrix: function( transformMatrix ) {
      this.transformMatrix = transformMatrix;
      return this;
    },
    
    getCanvasStyle: function() {
      return this.canvasGradient;
    },
    
    // seems we need the defs: http://stackoverflow.com/questions/7614209/linear-gradients-in-svg-without-defs
    // SVG: spreadMethod 'pad' 'reflect' 'repeat' - find Canvas usage
    getSVGDefinition: function( id ) {
      /* Approximate example of what we are creating:
      <linearGradient id="grad2" x1="0" y1="0" x2="100" y2="0" gradientUnits="userSpaceOnUse">
        <stop offset="0" style="stop-color:rgb(255,255,0);stop-opacity:1" />
        <stop offset="0.5" style="stop-color:rgba(255,255,0,0);stop-opacity:0" />
        <stop offset="1" style="stop-color:rgb(255,0,0);stop-opacity:1" />
      </linearGradient>
      */
      var svgns = 'http://www.w3.org/2000/svg'; // TODO: store this in a common place!
      var definition = document.createElementNS( svgns, 'linearGradient' );
      definition.setAttribute( 'id', id );
      definition.setAttribute( 'gradientUnits', 'userSpaceOnUse' ); // so we don't depend on the bounds of the object being drawn with the gradient
      definition.setAttribute( 'x1', this.start.x );
      definition.setAttribute( 'y1', this.start.y );
      definition.setAttribute( 'x2', this.end.x );
      definition.setAttribute( 'y2', this.end.y );
      if ( this.transformMatrix ) {
        definition.setAttribute( 'gradientTransform', this.transformMatrix.getSVGTransform() );
      }
      
      _.each( this.stops, function( stop ) {
        // TODO: store color in our stops array, so we don't have to create additional objects every time?
        var color = new scenery.Color( stop.color );
        var stopElement = document.createElementNS( svgns, 'stop' );
        stopElement.setAttribute( 'offset', stop.ratio );
        stopElement.setAttribute( 'style', 'stop-color: ' + color.withAlpha( 1 ).getCSS() + '; stop-opacity: ' + color.a.toFixed( 20 ) + ';' );
        definition.appendChild( stopElement );
      } );
      
      return definition;
    },
    
    toString: function() {
      var result = 'new scenery.LinearGradient( ' + this.start.x + ', ' + this.start.y + ', ' + this.end.x + ', ' + this.end.y + ' )';
      
      _.each( this.stops, function( stop ) {
        result += '.addColorStop( ' + stop.ratio + ', \'' + stop.color + '\' )';
      } );
      
      return result;
    }
  };
  
  return LinearGradient;
} );

// Copyright 2002-2012, University of Colorado

/**
 * A pattern that will deliver a fill or stroke that will repeat an image in both directions (x and y).
 *
 * TODO: future support for repeat-x, repeat-y or no-repeat (needs SVG support)
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/util/Pattern',['require','ASSERT/assert','SCENERY/scenery','DOT/Vector2'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Vector2 = require( 'DOT/Vector2' );
  
  // TODO: support scene or other various content (SVG is flexible, can backport to canvas)
  // TODO: investigate options to support repeat-x, repeat-y or no-repeat in SVG (available repeat options from Canvas)
  scenery.Pattern = function( image ) {
    this.image = image;
    
    // TODO: make a global spot that will have a 'useless' context for these purposes?
    this.canvasPattern = document.createElement( 'canvas' ).getContext( '2d' ).createPattern( image, 'repeat' );
    
    this.transformMatrix = null;
  };
  var Pattern = scenery.Pattern;
  
  Pattern.prototype = {
    constructor: Pattern,
    
    setTransformMatrix: function( transformMatrix ) {
      this.transformMatrix = transformMatrix;
      return this;
    },
    
    getCanvasStyle: function() {
      return this.canvasPattern;
    },
    
    getSVGDefinition: function( id ) {
      var svgns = 'http://www.w3.org/2000/svg'; // TODO: store this in a common place!
      var definition = document.createElementNS( svgns, 'pattern' );
      definition.setAttribute( 'id', id );
      definition.setAttribute( 'patternUnits', 'userSpaceOnUse' ); // so we don't depend on the bounds of the object being drawn with the gradient
      definition.setAttribute( 'patternContentUnits', 'userSpaceOnUse' ); // TODO: is this needed?
      definition.setAttribute( 'x', 0 );
      definition.setAttribute( 'y', 0 );
      definition.setAttribute( 'width', this.image.width );
      definition.setAttribute( 'height', this.image.height );
      if ( this.transformMatrix ) {
        definition.setAttribute( 'patternTransform', this.transformMatrix.getSVGTransform() );
      }
      
      definition.appendChild( scenery.Image.createSVGImage( this.image.src, this.image.width, this.image.height ) );
      
      return definition;
    },
    
    toString: function() {
      return 'new scenery.Pattern( $( \'<img src="' + this.image.src + '"/>\' )[0] )';
    }
  };
  
  return Pattern;
} );

// Copyright 2002-2012, University of Colorado

/**
 * A radial gradient that can be passed into the 'fill' or 'stroke' parameters.
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/util/RadialGradient',['require','ASSERT/assert','SCENERY/scenery','DOT/Vector2'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Vector2 = require( 'DOT/Vector2' );
  
  // TODO: support Vector2s for p0 and p1
  scenery.RadialGradient = function( x0, y0, r0, x1, y1, r1 ) {
    this.start = new Vector2( x0, y0 );
    this.end = new Vector2( x1, y1 );
    this.startRadius = r0;
    this.endRadius = r1;
    
    // linear function from radius to point on the line from start to end
    this.focalPoint = this.start.plus( this.end.minus( this.start ).times( this.startRadius / ( this.startRadius - this.endRadius ) ) );
    
    // make sure that the focal point is in both circles. SVG doesn't support rendering outside of them
    if ( this.startRadius >= this.endRadius ) {
      assert && assert( this.focalPoint.minus( this.start ).magnitude() <= this.startRadius );
    } else {
      assert && assert( this.focalPoint.minus( this.end ).magnitude() <= this.endRadius );
    }
    
    this.stops = [];
    this.lastStopRatio = 0;
    
    // TODO: make a global spot that will have a 'useless' context for these purposes?
    this.canvasGradient = document.createElement( 'canvas' ).getContext( '2d' ).createRadialGradient( x0, y0, r0, x1, y1, r1 );
    
    this.transformMatrix = null;
  };
  var RadialGradient = scenery.RadialGradient;
  
  RadialGradient.prototype = {
    constructor: RadialGradient,
    
    addColorStop: function( ratio, color ) {
      if ( this.lastStopRatio > ratio ) {
        // fail out, since browser quirks go crazy for this case
        throw new Error( 'Color stops not specified in the order of increasing ratios' );
      } else {
        this.lastStopRatio = ratio;
      }
      
      this.stops.push( { ratio: ratio, color: color } );
      this.canvasGradient.addColorStop( ratio, color );
      return this;
    },
    
    setTransformMatrix: function( transformMatrix ) {
      this.transformMatrix = transformMatrix;
      return this;
    },
    
    getCanvasStyle: function() {
      return this.canvasGradient;
    },
    
    getSVGDefinition: function( id ) {
      var startIsLarger = this.startRadius > this.endRadius;
      var largePoint = startIsLarger ? this.start : this.end;
      var smallPoint = startIsLarger ? this.end : this.start;
      var maxRadius = Math.max( this.startRadius, this.endRadius );
      var minRadius = Math.min( this.startRadius, this.endRadius );
      
      var svgns = 'http://www.w3.org/2000/svg'; // TODO: store this in a common place!
      var definition = document.createElementNS( svgns, 'radialGradient' );
      
      // TODO:
      definition.setAttribute( 'id', id );
      definition.setAttribute( 'gradientUnits', 'userSpaceOnUse' ); // so we don't depend on the bounds of the object being drawn with the gradient
      definition.setAttribute( 'cx', largePoint.x );
      definition.setAttribute( 'cy', largePoint.y );
      definition.setAttribute( 'r', maxRadius );
      definition.setAttribute( 'fx', this.focalPoint.x );
      definition.setAttribute( 'fy', this.focalPoint.y );
      if ( this.transformMatrix ) {
        definition.setAttribute( 'gradientTransform', this.transformMatrix.getSVGTransform() );
      }
      
      // maps x linearly from [a0,b0] => [a1,b1]
      function linearMap( a0, b0, a1, b1, x ) {
        return a1 + ( x - a0 ) * ( b1 - a1 ) / ( b0 - a0 );
      }
      
      function applyStop( stop ) {
        // flip the stops if the start has a larger radius
        var ratio = startIsLarger ? 1 - stop.ratio : stop.ratio;
        
        // scale the stops properly if the smaller radius isn't 0
        if ( minRadius > 0 ) {
          // scales our ratio from [0,1] => [minRadius/maxRadius,0]
          ratio = linearMap( 0, 1, minRadius / maxRadius, 1, ratio );
        }
        
        // TODO: store color in our stops array, so we don't have to create additional objects every time?
        var color = new scenery.Color( stop.color );
        var stopElement = document.createElementNS( svgns, 'stop' );
        stopElement.setAttribute( 'offset', ratio );
        stopElement.setAttribute( 'style', 'stop-color: ' + color.withAlpha( 1 ).getCSS() + '; stop-opacity: ' + color.a.toFixed( 20 ) + ';' );
        definition.appendChild( stopElement );
      }
      
      var i;
      // switch the direction we apply stops in, so that the ratios always are increasing.
      if ( startIsLarger ) {
        for ( i = this.stops.length - 1; i >= 0; i-- ) {
          applyStop( this.stops[i] );
        }
      } else {
        for ( i = 0; i < this.stops.length; i++ ) {
          applyStop( this.stops[i] );
        }
      }
      
      return definition;
    },
    
    toString: function() {
      var result = 'new scenery.RadialGradient( ' + this.start.x + ', ' + this.start.y + ', ' + this.startRadius + ', ' + this.end.x + ', ' + this.end.y + ', ' + this.endRadius + ' )';
      
      _.each( this.stops, function( stop ) {
        result += '.addColorStop( ' + stop.ratio + ', \'' + stop.color + '\' )';
      } );
      
      return result;
    }
  };
  
  return RadialGradient;
} );

// Copyright 2002-2012, University of Colorado

/*
 * An HTMLImageElement that is backed by a scene. Call update() on this SceneImage to update the image from the scene.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/util/SceneImage',['require','ASSERT/assert','SCENERY/scenery'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  // NOTE: ideally the scene shouldn't use SVG, since rendering that to a canvas takes a callback (and usually requires canvg)
  scenery.SceneImage = function( scene ) {
    this.scene = scene;
    
    // we write the scene to a canvas, get its data URL, and pass that to the image.
    this.canvas = document.createElement( 'canvas' );
    this.context = this.canvas.getContext( '2d' );
    
    this.img = document.createElement( 'img' );
    this.update();
  };
  var SceneImage = scenery.SceneImage;
  
  SceneImage.prototype = {
    constructor: SceneImage,
    
    // NOTE: calling this before the previous update() completes may cause the previous onComplete to not be executed
    update: function( onComplete ) {
      var self = this;
      
      this.scene.updateScene();
      
      this.canvas.width = this.scene.getSceneWidth();
      this.canvas.height = this.scene.getSceneHeight();
      
      this.scene.renderToCanvas( this.canvas, this.context, function() {
        var url = self.toDataURL();
        
        self.img.onload = function() {
          onComplete();
          delete self.img.onload;
        };
        self.img.src = url;
      } );
    }
  };
  
  return SceneImage;
} );

// Copyright 2002-2012, University of Colorado

/**
 * An interval between two Trails. A trail being null means either 'from the start' or 'to the end', depending
 * on whether it is the first or second parameter to the constructor.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/util/TrailInterval',['require','ASSERT/assert','SCENERY/scenery','SCENERY/util/Trail'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );
  
  // dataA and dataB are arbitrary types of data that can be attached, and are preserved on combination operations
  scenery.TrailInterval = function( a, b, dataA, dataB ) {
    assert && assert( !a || !b || a.compare( b ) <= 0, 'TrailInterval parameters must not be out of order' );
    
    this.a = a;
    this.b = b;
    
    // data associated to each endpoint of the interval
    this.dataA = dataA;
    this.dataB = dataB;
  };
  var TrailInterval = scenery.TrailInterval;
  
  TrailInterval.prototype = {
    constructor: TrailInterval,
    
    reindex: function() {
      this.a && this.a.reindex();
      this.b && this.b.reindex();
    },
    
    isValidExclusive: function() {
      // like construction, but with strict inequality
      return !this.a || !this.b || this.a.compare( this.b ) < 0;
    },
    
    /*
     * Whether the union of this and the specified interval doesn't include any additional trails, when
     * both are treated as exclusive endpoints (exclusive between a and b). We also make the assumption
     * that a !== b || a === null for either interval, since otherwise it is not well defined.
     */
    exclusiveUnionable: function( interval ) {
      assert && assert ( this.isValidExclusive(), 'exclusiveUnionable requires exclusive intervals' );
      assert && assert ( interval.isValidExclusive(), 'exclusiveUnionable requires exclusive intervals' );
      return ( !this.a || !interval.b || this.a.compare( interval.b ) === -1 ) &&
             ( !this.b || !interval.a || this.b.compare( interval.a ) === 1 );
    },
    
    exclusiveContains: function( trail ) {
      assert && assert( trail );
      return ( !this.a || this.a.compare( trail ) < 0 ) && ( !this.b || this.b.compare( trail ) > 0 );
    },
    
    union: function( interval ) {
      // falsy checks since if a or b is null, we want that bound to be null
      var thisA = ( !this.a || ( interval.a && this.a.compare( interval.a ) === -1 ) );
      var thisB = ( !this.b || ( interval.b && this.b.compare( interval.b ) === 1 ) );
      
      return new TrailInterval(
        thisA ? this.a : interval.a,
        thisB ? this.b : interval.b,
        thisA ? this.dataA : interval.dataA,
        thisB ? this.dataB : interval.dataB
      );
    },
    
    toString: function() {
      return '[' + ( this.a ? this.a.toString() : this.a ) + ',' + ( this.b ? this.b.toString() : this.b ) + ']';
    }
  };
  
  return TrailInterval;
} );



// Copyright 2002-2012, University of Colorado

/**
 * Main scene, that is also a Node.
 *
 * TODO: documentation!
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('SCENERY/Scene',['require','ASSERT/assert','DOT/Bounds2','DOT/Vector2','DOT/Matrix3','SCENERY/scenery','SCENERY/nodes/Node','SCENERY/util/Trail','SCENERY/util/TrailInterval','SCENERY/util/TrailPointer','SCENERY/input/Input','SCENERY/layers/LayerBuilder','SCENERY/layers/Renderer','SCENERY/util/Util'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Vector2 = require( 'DOT/Vector2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' ); // inherits from Node
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/util/TrailInterval' );
  require( 'SCENERY/util/TrailPointer' );
  require( 'SCENERY/input/Input' );
  require( 'SCENERY/layers/LayerBuilder' );
  require( 'SCENERY/layers/Renderer' );
  
  var Util = require( 'SCENERY/util/Util' );
  var objectCreate = Util.objectCreate;
  
  // if assertions are enabled, log out layer information
  var layerLogger = null; //assert ? function( ob ) { console.log( ob ); } : null;
  
  /*
   * $main should be a block-level element with a defined width and height. scene.resize() should be called whenever
   * it is resized.
   *
   * Valid parameters in the parameter object:
   * {
   *   allowSceneOverflow: false,           // usually anything displayed outside of this $main (DOM/CSS3 transformed SVG) is hidden with CSS overflow
   *   allowCSSHacks: true,                 // applies styling that prevents mobile browser graphical issues
   *   allowDevicePixelRatioScaling: false, // allows underlying canvases (Canvas, WebGL) to increase in size to maintain sharpness on high-density displays
   *   enablePointerEvents: true,           // allows pointer events / MSPointerEvent to be used on supported platforms.
   *   preferredSceneLayerType: ...,        // sets the preferred type of layer to be created if there are multiple options
   *   width: <current main width>,         // override the main container's width
   *   height: <current main height>,       // override the main container's height
   * }
   */
  scenery.Scene = function Scene( $main, options ) {
    assert && assert( $main[0], 'A main container is required for a scene' );
    
    // defaults
    options = _.extend( {
      allowSceneOverflow: false,
      allowCSSHacks: true,
      allowDevicePixelRatioScaling: false,
      enablePointerEvents: true,
      preferredSceneLayerType: scenery.CanvasDefaultLayerType,
      width: $main.width(),
      height: $main.height()
    }, options || {} );
    
    // TODO: consider using a pushed preferred layer to indicate this information, instead of as a specific option
    this.backingScale = options.allowDevicePixelRatioScaling ? Util.backingScale( document.createElement( 'canvas' ).getContext( '2d' ) ) : 1;
    this.enablePointerEvents = options.enablePointerEvents;
    
    Node.call( this, options );
    
    var scene = this;
    window.debugScene = scene;
    
    // main layers in a scene
    this.layers = [];
    
    this.layerChangeIntervals = []; // array of {TrailInterval}s indicating what parts need to be stitched together
    
    this.lastCursor = null;
    this.defaultCursor = $main.css( 'cursor' );
    
    this.$main = $main;
    // resize the main container as a sanity check
    this.setSize( options.width, options.height );
    
    this.sceneBounds = new Bounds2( 0, 0, $main.width(), $main.height() );
    
    // default to a canvas layer type, but this can be changed
    this.preferredSceneLayerType = options.preferredSceneLayerType;
    
    applyCSSHacks( $main, options );
    
    // note, arguments to the functions are mutable. don't destroy them
    this.sceneEventListener = {
      markForLayerRefresh: function( args ) { // contains trail
        layerLogger && layerLogger( 'marking layer refresh: ' + args.trail.toString() );
        scene.markInterval( args.trail );
      },
      
      markForInsertion: function( args ) { // contains parent, child, index, trail
        var affectedTrail = args.trail.copy().addDescendant( args.child );
        layerLogger && layerLogger( 'marking insertion: ' + affectedTrail.toString() );
        scene.markInterval( affectedTrail );
      },
      
      markForRemoval: function( args ) { // contains parent, child, index, trail
        // mark the interval
        var affectedTrail = args.trail.copy().addDescendant( args.child );
        layerLogger && layerLogger( 'marking removal: ' + affectedTrail.toString() );
        scene.markInterval( affectedTrail );
        
        // signal to the relevant layers to remove the specified trail while the trail is still valid.
        // waiting until after the removal takes place would require more complicated code to properly handle the trails
        affectedTrail.eachTrailUnder( function( trail ) {
          if ( trail.isPainted() ) {
            scene.layerLookup( trail ).removeNodeFromTrail( trail );
          }
        } );
      },
      
      stitch: function( args ) { // contains match {Boolean}
        scene.stitch( args.match );
      },
      
      dirtyBounds: function( args ) { // contains node, bounds, transform, trail
        var trail = args.trail;
        
        // if there are no layers, no nodes would actually render, so don't do the lookup
        if ( scene.layers.length ) {
          _.each( scene.affectedLayers( trail ), function( layer ) {
            layer.markDirtyRegion( args );
          } );
        }
      },
      
      transform: function( args ) { // conatins node, type, matrix, transform, trail
        var trail = args.trail;
        
        if ( scene.layers.length ) {
          _.each( scene.affectedLayers( trail ), function( layer ) {
            layer.transformChange( args );
          } );
        }
      }
    };
    
    this.addEventListener( this.sceneEventListener );
  };
  var Scene = scenery.Scene;

  Scene.prototype = objectCreate( Node.prototype );
  Scene.prototype.constructor = Scene;
  
  Scene.prototype.updateScene = function( args ) {
    // validating bounds, similar to Piccolo2d
    this.validateBounds();
    this.validatePaint();
    
    // bail if there are no layers. consider a warning?
    if ( !this.layers.length ) {
      return;
    }
    
    var scene = this;
    
    _.each( this.layers, function( layer ) {
      layer.render( scene, args );
    } );
    
    this.updateCursor();
  };
  
  Scene.prototype.renderScene = function() {
    // TODO: for now, go with the same path. possibly add options later
    this.updateScene();
  };
  
  Scene.prototype.markInterval = function( affectedTrail ) {
    // since this is marked while the child is still connected, we can use our normal trail handling.
    
    // find the closest before and after self trails that are not affected
    var beforeTrail = affectedTrail.previousPainted(); // easy for the before trail
    
    var afterTrailPointer = new scenery.TrailPointer( affectedTrail.copy(), false );
    while ( afterTrailPointer.hasTrail() && ( !afterTrailPointer.isBefore || !afterTrailPointer.trail.isPainted() ) ) {
      afterTrailPointer.nestedForwards();
    }
    var afterTrail = afterTrailPointer.trail;
    
    // store the layer of the before/after trails so that it is easy to access later
    this.addLayerChangeInterval( new scenery.TrailInterval(
      beforeTrail,
      afterTrail,
      beforeTrail ? this.layerLookup( beforeTrail ) : null,
      afterTrail ? this.layerLookup( afterTrail ) : null
    ) );
  };
  
  // convenience function for layer change intervals
  Scene.prototype.addLayerChangeInterval = function( interval ) {
    layerLogger && layerLogger( 'adding interval: ' + interval.toString() );
    // TODO: replace with a binary-search-like version that may be faster. this includes a full scan
    
    // attempt to merge this interval with another if possible.
    for ( var i = 0; i < this.layerChangeIntervals.length; i++ ) {
      var other = this.layerChangeIntervals[i];
      other.reindex(); // sanity check, although for most use-cases this should be unnecessary
      
      if ( interval.exclusiveUnionable( other ) ) {
        // the interval can be unioned without including other nodes. do this, and remove the other interval from consideration
        interval = interval.union( other );
        this.layerChangeIntervals.splice( i, 1 );
      }
    }
    
    this.layerChangeIntervals.push( interval );
  };
  
  Scene.prototype.createLayer = function( layerType, layerArgs, startBoundary, endBoundary ) {
    var layer = layerType.createLayer( _.extend( {
      startBoundary: startBoundary,
      endBoundary: endBoundary
    }, layerArgs ) );
    layer.type = layerType;
    layerLogger && layerLogger( 'created layer: ' + layer.getId() + ' of type ' + layer.type.name );
    return layer;
  };
  
  // insert a layer into the proper place (from its starting boundary)
  Scene.prototype.insertLayer = function( layer ) {
    if ( this.layers.length > 0 && this.layers[0].startBoundary.equivalentPreviousTrail( layer.endBoundary.previousPaintedTrail ) ) {
      // layer needs to be inserted at the very beginning
      this.layers.unshift( layer );
    } else {
      for ( var i = 0; i < this.layers.length; i++ ) {
        // compare end and start boundaries, as they should match
        if ( this.layers[i].endBoundary.equivalentNextTrail( layer.startBoundary.nextPaintedTrail ) ) {
          break;
        }
      }
      if ( i < this.layers.length ) {
        this.layers.splice( i + 1, 0, layer );
      } else {
        this.layers.push( layer );
      }
    }
  };
  
  Scene.prototype.calculateBoundaries = function( beforeLayerType, beforeTrail, afterTrail ) {
    layerLogger && layerLogger( 'build between ' + ( beforeTrail ? beforeTrail.toString() : beforeTrail ) + ',' + ( afterTrail ? afterTrail.toString() : afterTrail ) + ' with beforeType: ' + ( beforeLayerType ? beforeLayerType.name : null ) );
    var builder = new scenery.LayerBuilder( this, beforeLayerType, beforeTrail, afterTrail );
    
    // push the preferred layer type before we push that for any nodes
    if ( this.preferredSceneLayerType ) {
      builder.pushPreferredLayerType( this.preferredSceneLayerType );
    }
    
    builder.run();
    
    return builder.boundaries;
  };
  
  Scene.prototype.stitch = function( match ) {
    var scene = this;
    
    // we need to map old layer IDs to new layers if we 'glue' two layers into one, so that the layer references we put on the
    // intervals can be mapped to current layers.
    var layerMap = {};
    
    // default arguments for constructing layers
    var layerArgs = {
      $main: this.$main,
      scene: this,
      baseNode: this
    };
    
    /*
     * Sort our intervals, so that when we need to 'unglue' a layer into two separate layers, we will have passed
     * all of the parts where we would need to use the 'before' layer, so we can update our layer map with the 'after'
     * layer.
     */
    this.layerChangeIntervals.sort( function( a, b ) {
      // TODO: consider TrailInterval parameter renaming
      return a.a.compare( b.a );
    } );
    
    layerLogger && layerLogger( 'stitching on intervals: \n' + this.layerChangeIntervals.join( '\n' ) );
    
    _.each( this.layerChangeIntervals, function( interval ) {
      layerLogger && layerLogger( 'before reindex: ' + interval.toString() );
      interval.reindex();
      layerLogger && layerLogger( 'stitch on interval ' + interval.toString() );
      var beforeTrail = interval.a;
      var afterTrail = interval.b;
      
      // stored here, from in markInterval
      var beforeLayer = interval.dataA;
      var afterLayer = interval.dataB;
      
      // if these layers are out of date, update them. 'while' will handle chained updates. circular references should be impossible
      while ( beforeLayer && layerMap[beforeLayer.getId()] ) {
        beforeLayer = layerMap[beforeLayer.getId()];
      }
      while ( afterLayer && layerMap[afterLayer.getId()] ) {
        afterLayer = layerMap[afterLayer.getId()];
      }
      
      var boundaries = scene.calculateBoundaries( beforeLayer ? beforeLayer.type : null, beforeTrail, afterTrail );
      
      // if ( match ) {
        // TODO: patch in the matching version!
        // scene.rebuildLayers(); // bleh
      // } else {
      scene.stitchInterval( layerMap, layerArgs, beforeTrail, afterTrail, beforeLayer, afterLayer, boundaries, match );
      // }
    } );
    this.layerChangeIntervals = [];
    
    this.reindexLayers();
    
    // TODO: add this back in, but with an appropriate assertion level
    // assert && assert( this.layerAudit() );
  };
  
  /*
   * Stitching intervals has essentially two specific modes:
   * non-matching: handles added or removed nodes (and this can span multiple, even adjacent trails)
   * matching: handles in-place layer refreshes (no nodes removed or added, but something like a renderer was changed)
   *
   * This separation occurs since for matching, we want to match old layers with possible new layers, so we can keep trails in their
   * current layer instead of creating an identical layer and moving the trails to that layer.
   *
   * The stitching basically re-does the layering between a start and end trail, attempting to minimize the amount of changes made.
   * It can include 'gluing' layers together (a node that caused layer splits was removed, and before/after layers are joined),
   * 'ungluing' layers (an inserted node causes a layer split in an existing layer, and it is separated into a before/after),
   * or normal updating of the interior.
   *
   * The beforeTrail and afterTrail should be outside the modifications, and if the modifications are to the start/end of the graph,
   * they should be passed as null to indicate 'before everything' and 'after everything' respectively.
   *
   * Here be dragons!
   */
  Scene.prototype.stitchInterval = function( layerMap, layerArgs, beforeTrail, afterTrail, beforeLayer, afterLayer, boundaries, match ) {
    var scene = this;
    
    // need a reference to this, since it may changes
    var afterLayerEndBoundary = afterLayer ? afterLayer.endBoundary : null;
    
    var beforeLayerIndex = beforeLayer ? _.indexOf( this.layers, beforeLayer ) : -1;
    var afterLayerIndex = afterLayer ? _.indexOf( this.layers, afterLayer ) : this.layers.length;
    
    var beforePointer = beforeTrail ? new scenery.TrailPointer( beforeTrail, true ) : new scenery.TrailPointer( new scenery.Trail( this ), true );
    var afterPointer = afterTrail ? new scenery.TrailPointer( afterTrail, true ) : new scenery.TrailPointer( new scenery.Trail( this ), false );
    
    layerLogger && layerLogger( 'stitching with boundaries:\n' + _.map( boundaries, function( boundary ) { return boundary.toString(); } ).join( '\n' ) );
    
    // maps trail unique ID => layer, only necessary when matching since we need to remove trails from their old layers
    var oldLayerMap = match ? this.mapTrailLayersBetween( beforeTrail, afterTrail ) : null;
    
    /*---------------------------------------------------------------------------*
    * State
    *----------------------------------------------------------------------------*/
    
    var nextBoundaryIndex = 0;
    var nextBoundary = boundaries[nextBoundaryIndex];
    var trailsToAddToLayer = [];
    var currentTrail = beforeTrail;
    var currentLayer = beforeLayer;
    var currentLayerType = beforeLayer ? beforeLayer.type : null;
    var currentStartBoundary = null;
    var matchingLayer = null; // set whenever a trail has a matching layer, cleared after boundary
    
    var layersToAdd = [];
    
    // a list of layers that are most likely removed, not including the afterLayer for gluing
    var layersToRemove = [];
    for ( var i = beforeLayerIndex + 1; i < afterLayerIndex; i++ ) {
      layersToRemove.push( this.layers[i] );
    }
    
    function addPendingTrailsToLayer() {
      // add the necessary nodes to the layer
      _.each( trailsToAddToLayer, function( trail ) {
        if ( match ) {
          // only remove/add if the layer has actually changed. if we are preserving the layer, don't do anything
          var oldLayer = oldLayerMap[trail.getUniqueId()];
          if ( oldLayer !== currentLayer ) {
            oldLayer.removeNodeFromTrail( trail );
            currentLayer.addNodeFromTrail( trail );
          }
        } else {
          currentLayer.addNodeFromTrail( trail );
        }
      } );
      trailsToAddToLayer = [];
    }
    
    function addLayerForRemoval( layer ) {
      if ( !_.contains( layersToRemove, layer ) ) {
        layersToRemove.push( afterLayer );
      }
    }
    
    function addAndCreateLayer( startBoundary, endBoundary ) {
      currentLayer = scene.createLayer( currentLayerType, layerArgs, startBoundary, endBoundary );
      layersToAdd.push( currentLayer );
    }
    
    function step( trail, isEnd ) {
      layerLogger && layerLogger( 'step: ' + ( trail ? trail.toString() : trail ) );
      // check for a boundary at this step between currentTrail and trail
      
      // if there is no next boundary, don't bother checking anyways
      if ( nextBoundary && nextBoundary.equivalentPreviousTrail( currentTrail ) ) { // at least one null check
        assert && assert( nextBoundary.equivalentNextTrail( trail ) );
        
        layerLogger && layerLogger( nextBoundary.toString() );
        
        // we are at a boundary change. verify that we are at the end of a layer
        if ( currentLayer || currentStartBoundary ) {
          if ( currentLayer ) {
            layerLogger && layerLogger( 'has currentLayer: ' + currentLayer.getId() );
            // existing layer, reposition its endpoint
            currentLayer.setEndBoundary( nextBoundary );
          } else {
            assert && assert( currentStartBoundary );
            
            if ( matchingLayer ) {
              layerLogger && layerLogger( 'matching layer used: ' + matchingLayer.getId() );
              matchingLayer.setStartBoundary( currentStartBoundary );
              matchingLayer.setEndBoundary( nextBoundary );
              currentLayer = matchingLayer;
            } else {
              layerLogger && layerLogger( 'creating layer' );
              addAndCreateLayer( currentStartBoundary, nextBoundary ); // sets currentLayer
            }
          }
          // sanity checks
          assert && assert( currentLayer.startPaintedTrail );
          assert && assert( currentLayer.endPaintedTrail );
          
          addPendingTrailsToLayer();
        } else {
          // if not at the end of a layer, sanity check that we should have no accumulated pending trails
          layerLogger && layerLogger( 'was first layer' );
          assert && assert( trailsToAddToLayer.length === 0 );
        }
        currentLayer = null;
        currentLayerType = nextBoundary.nextLayerType;
        currentStartBoundary = nextBoundary;
        matchingLayer = null;
        nextBoundaryIndex++;
        nextBoundary = boundaries[nextBoundaryIndex];
      }
      if ( trail && !isEnd ) {
        trailsToAddToLayer.push( trail );
      }
      if ( match && !isEnd ) { // TODO: verify this condition with test cases
        // if the node's old layer is compatible
        var layer = oldLayerMap[trail.getUniqueId()];
        if ( layer.type === currentLayerType ) {
          matchingLayer = layer;
        }
      }
      currentTrail = trail;
    }
    
    function startStep( trail ) {
      layerLogger && layerLogger( 'startStep: ' + ( trail ? trail.toString() : trail ) );
    }
    
    function middleStep( trail ) {
      layerLogger && layerLogger( 'middleStep: ' + trail.toString() );
      step( trail, false );
    }
    
    function endStep( trail ) {
      layerLogger && layerLogger( 'endStep: ' + ( trail ? trail.toString() : trail ) );
      step( trail, true );
      
      if ( beforeLayer !== afterLayer && boundaries.length === 0 ) {
        // glue the layers together
        layerLogger && layerLogger( 'gluing layer' );
        layerLogger && layerLogger( 'endBoundary: ' + afterLayer.endBoundary.toString() );
        beforeLayer.setEndBoundary( afterLayer.endBoundary );
        addLayerForRemoval( afterLayer );
        currentLayer = beforeLayer;
        addPendingTrailsToLayer();
        
        // move over all of afterLayer's trails to beforeLayer
        // defensive copy needed, since this will be modified at the same time
        _.each( afterLayer._layerTrails.slice( 0 ), function( trail ) {
          trail.reindex();
          afterLayer.removeNodeFromTrail( trail );
          beforeLayer.addNodeFromTrail( trail );
        } );
        
        layerMap[afterLayer.getId()] = beforeLayer;
      } else if ( beforeLayer && beforeLayer === afterLayer && boundaries.length > 0 ) {
        // need to 'unglue' and split the layer
        layerLogger && layerLogger( 'ungluing layer' );
        assert && assert( currentStartBoundary );
        addAndCreateLayer( currentStartBoundary, afterLayerEndBoundary ); // sets currentLayer
        layerMap[afterLayer.getId()] = currentLayer;
        addPendingTrailsToLayer();
        
        scenery.Trail.eachPaintedTrailbetween( afterTrail, currentLayer.endPaintedTrail, function( trail ) {
          trail.reindex();
          afterLayer.removeNodeFromTrail( trail );
          currentLayer.addNodeFromTrail( trail );
        }, false, scene );
      } else if ( !beforeLayer && !afterLayer && boundaries.length === 1 && !boundaries[0].hasNext() && !boundaries[0].hasPrevious() ) {
        // TODO: why are we generating a boundary here?!?
      } else {
        currentLayer = afterLayer;
        // TODO: check concepts on this guard, since it seems sketchy
        if ( currentLayer && currentStartBoundary ) {
          currentLayer.setStartBoundary( currentStartBoundary );
        }
        
        addPendingTrailsToLayer();
      }
      
      _.each( layersToRemove, function( layer ) {
        layerLogger && layerLogger( 'disposing layer: ' + layer.getId() );
        scene.disposeLayer( layer );
      } );
      _.each( layersToAdd, function( layer ) {
        layerLogger && layerLogger( 'inserting layer: ' + layer.getId() );
        scene.insertLayer( layer );
      } );
    }
    
    // iterate from beforeTrail up to BEFORE the afterTrail. does not include afterTrail
    startStep( beforeTrail );
    beforePointer.eachTrailBetween( afterPointer, function( trail ) {
      // ignore non-self trails
      if ( !trail.isPainted() || ( beforeTrail && trail.equals( beforeTrail ) ) ) {
        return;
      }
      
      middleStep( trail.copy() );
    } );
    endStep( afterTrail );
  };
  
  // returns a map from trail.getUniqueId() to the current layer in which that trail resides
  Scene.prototype.mapTrailLayersBetween = function( beforeTrail, afterTrail ) {
    var scene = this;
    
    var result = {};
    scenery.Trail.eachPaintedTrailbetween( beforeTrail, afterTrail, function( trail ) {
      // TODO: optimize this! currently both the layer lookup and this inefficient method of using layer lookup is slow
      result[trail.getUniqueId()] = scene.layerLookup( trail );
    }, false, this );
    
    return result;
  };
  
  Scene.prototype.rebuildLayers = function() {
    layerLogger && layerLogger( 'rebuildLayers' );
    // remove all of our tracked layers from the container, so we can fill it with fresh layers
    this.disposeLayers();
    
    this.boundaries = this.calculateBoundaries( null, null, null );
    
    var layerArgs = {
      $main: this.$main,
      scene: this,
      baseNode: this
    };
    
    this.layers = [];
    
    layerLogger && layerLogger( this.boundaries );
    
    for ( var i = 1; i < this.boundaries.length; i++ ) {
      var startBoundary = this.boundaries[i-1];
      var endBoundary = this.boundaries[i];
      
      assert && assert( startBoundary.nextLayerType === endBoundary.previousLayerType );
      var layerType = startBoundary.nextLayerType;
      
      // LayerType is responsible for applying its own arguments in createLayer()
      var layer = layerType.createLayer( _.extend( {
        startBoundary: startBoundary,
        endBoundary: endBoundary
      }, layerArgs ) );
      
      // record the type on the layer
      layer.type = layerType;
      
      // add the initial nodes to the layer
      layer.startPointer.eachTrailBetween( layer.endPointer, function( trail ) {
        if ( trail.isPainted() ) {
          layer.addNodeFromTrail( trail );
        }
      } );
      
      this.layers.push( layer );
    }
  };
  
  // after layer changes, the layers should have their zIndex updated
  Scene.prototype.reindexLayers = function() {
    var index = 1;
    _.each( this.layers, function( layer ) {
      // layers increment indices as needed
      index = layer.reindex( index );
    } );
  };
  
  Scene.prototype.dispose = function() {
    this.disposeLayers();
    
    // TODO: clear event handlers if added
    //throw new Error( 'unimplemented dispose: clear event handlers if added' );
  };
  
  Scene.prototype.disposeLayer = function( layer ) {
    layer.dispose();
    this.layers.splice( _.indexOf( this.layers, layer ), 1 ); // TODO: better removal code!
  };
  
  Scene.prototype.disposeLayers = function() {
    var scene = this;
    
    _.each( this.layers.slice( 0 ), function( layer ) {
      scene.disposeLayer( layer );
    } );
  };
  
  // what layer does this trail's terminal node render in? returns null if the node is not contained in a layer
  Scene.prototype.layerLookup = function( trail ) {
    // TODO: add tree form for optimization! this is slower than necessary, it shouldn't be O(n)!
    assert && assert( !( trail.isEmpty() || trail.nodes[0] !== this ), 'layerLookup root matches' );
    assert && assert( trail.isPainted(), 'layerLookup only supports nodes with isPainted(), as this guarantees an unambiguous answer' );
    
    if ( this.layers.length === 0 ) {
      return null; // node not contained in a layer
    }
    
    for ( var i = 0; i < this.layers.length; i++ ) {
      var layer = this.layers[i];
      
      // trails may be stale, so we need to update their indices
      if ( layer.startPaintedTrail ) { layer.startPaintedTrail.reindex(); }
      if ( layer.endPaintedTrail ) { layer.endPaintedTrail.reindex(); }
      
      if ( !layer.endPaintedTrail || trail.compare( layer.endPaintedTrail ) <= 0 ) {
        if ( !layer.startPaintedTrail || trail.compare( layer.startPaintedTrail ) >= 0 ) {
          return layer;
        } else {
          return null; // node is not contained in a layer (it is before any existing layer)
        }
      }
    }
    
    return null; // node not contained in a layer (it is after any existing layer)
  };
  
  // all layers whose start or end points lie inclusively in the range from the trail's before and after
  Scene.prototype.affectedLayers = function( trail ) {
    // TODO: add tree form for optimization! this is slower than necessary, it shouldn't be O(n)!
    
    var result = [];
    
    assert && assert( !( trail.isEmpty() || trail.nodes[0] !== this ), 'layerLookup root matches' );
    
    if ( this.layers.length === 0 ) {
      throw new Error( 'no layers in the scene' );
    }
    
    // point to the beginning of the node, right before it would be rendered
    var startPointer = new scenery.TrailPointer( trail, true );
    var endPointer = new scenery.TrailPointer( trail, false );
    
    for ( var i = 0; i < this.layers.length; i++ ) {
      var layer = this.layers[i];
      
      var notBefore = endPointer.compareNested( new scenery.TrailPointer( layer.startPaintedTrail, true ) ) !== -1;
      var notAfter = startPointer.compareNested( new scenery.TrailPointer( layer.endPaintedTrail, true ) ) !== 1;
      
      if ( notBefore && notAfter ) {
        result.push( layer );
      }
    }
    
    return result;
  };
  
  // attempt to render everything currently visible in the scene to an external canvas. allows copying from canvas layers straight to the other canvas
  Scene.prototype.renderToCanvas = function( canvas, context, callback ) {
    var count = 0;
    var started = false; // flag guards against asynchronous tests that call back synchronously (immediate increment and decrement)
    var delayCounts = {
      increment: function() {
        count++;
      },
      
      decrement: function() {
        count--;
        if ( count === 0 && callback && started ) {
          callback();
        }
      }
    };
    
    context.clearRect( 0, 0, canvas.width, canvas.height );
    _.each( this.layers, function( layer ) {
      layer.renderToCanvas( canvas, context, delayCounts );
    } );
    
    if ( count === 0 ) {
      // no asynchronous layers, callback immediately
      if ( callback ) {
        callback();
      }
    } else {
      started = true;
    }
  };
  
  // TODO: consider SVG data URLs
  
  Scene.prototype.canvasDataURL = function( callback ) {
    this.canvasSnapshot( function( canvas ) {
      callback( canvas.toDataURL() );
    } );
  };
  
  // renders what it can into a Canvas (so far, Canvas and SVG layers work fine)
  Scene.prototype.canvasSnapshot = function( callback ) {
    var canvas = document.createElement( 'canvas' );
    canvas.width = this.sceneBounds.getWidth();
    canvas.height = this.sceneBounds.getHeight();
    
    var context = canvas.getContext( '2d' );
    this.renderToCanvas( canvas, context, function() {
      callback( canvas, context.getImageData( 0, 0, canvas.width, canvas.height ) );
    } );
  };
  
  Scene.prototype.setSize = function( width, height ) {
    // resize our main container
    this.$main.width( width );
    this.$main.height( height );
    
    // set the container's clipping so anything outside won't show up
    // TODO: verify this clipping doesn't reduce performance!
    this.$main.css( 'clip', 'rect(0px,' + width + 'px,' + height + 'px,0px)' );
    
    this.sceneBounds = new Bounds2( 0, 0, width, height );
  };
  
  Scene.prototype.resize = function( width, height ) {
    this.setSize( width, height );
    this.rebuildLayers(); // TODO: why?
  };
  
  Scene.prototype.getSceneWidth = function() {
    return this.sceneBounds.getWidth();
  };
  
  Scene.prototype.getSceneHeight = function() {
    return this.sceneBounds.getHeight();
  };
  
  Scene.prototype.updateCursor = function() {
    if ( this.input && this.input.mouse.point ) {
      var mouseTrail = this.trailUnderPoint( this.input.mouse.point );
      
      if ( mouseTrail ) {
        for ( var i = mouseTrail.length - 1; i >= 0; i-- ) {
          var cursor = mouseTrail.nodes[i].getCursor();
          
          if ( cursor ) {
            this.setSceneCursor( cursor );
            return;
          }
        }
      }
    }
    
    // fallback case
    this.setSceneCursor( this.defaultCursor );
  };
  
  Scene.prototype.setSceneCursor = function( cursor ) {
    if ( cursor !== this.lastCursor ) {
      this.lastCursor = cursor;
      this.$main.css( 'cursor', cursor );
    }
  };
  
  Scene.prototype.updateOnRequestAnimationFrame = function( element ) {
    var scene = this;
    (function step() {
      window.requestAnimationFrame( step, element );
      scene.updateScene();
    })();
  };
  
  Scene.prototype.initializeStandaloneEvents = function( parameters ) {
    // TODO extract similarity between standalone and fullscreen!
    var element = this.$main[0];
    this.initializeEvents( _.extend( {}, {
      listenerTarget: element,
      pointFromEvent: function( evt ) {
        var mainBounds = element.getBoundingClientRect();
        return new Vector2( evt.clientX - mainBounds.left, evt.clientY - mainBounds.top );
      }
    }, parameters ) );
  };
  
  Scene.prototype.initializeFullscreenEvents = function( parameters ) {
    var element = this.$main[0];
    this.initializeEvents( _.extend( {}, {
      listenerTarget: document,
      pointFromEvent: function( evt ) {
        var mainBounds = element.getBoundingClientRect();
        return new Vector2( evt.clientX - mainBounds.left, evt.clientY - mainBounds.top );
      }
    }, parameters ) );
  };
  
  Scene.prototype.initializeEvents = function( parameters ) {
    var scene = this;
    
    if ( scene.input ) {
      throw new Error( 'Attempt to attach events twice to the scene' );
    }
    
    // TODO: come up with more parameter names that have the same string length, so it looks creepier
    var pointFromEvent = parameters.pointFromEvent;
    var listenerTarget = parameters.listenerTarget;
    
    var input = new scenery.Input( scene, listenerTarget );
    scene.input = input;
    
    // maps the current MS pointer types onto the pointer spec
    function msPointerType( evt ) {
      if ( evt.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_TOUCH ) {
        return 'touch';
      } else if ( evt.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_PEN ) {
        return 'pen';
      } else if ( evt.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_MOUSE ) {
        return 'mouse';
      } else {
        return evt.pointerType; // hope for the best
      }
    }
    
    function forEachChangedTouch( evt, callback ) {
      for ( var i = 0; i < evt.changedTouches.length; i++ ) {
        // according to spec (http://www.w3.org/TR/touch-events/), this is not an Array, but a TouchList
        var touch = evt.changedTouches.item( i );
        
        callback( touch.identifier, pointFromEvent( touch ) );
      }
    }
    
    // TODO: massive boilerplate reduction! closures should help tons!

    var implementsPointerEvents = window.navigator && window.navigator.pointerEnabled; // W3C spec for pointer events
    var implementsMSPointerEvents = window.navigator && window.navigator.msPointerEnabled; // MS spec for pointer event
    if ( this.enablePointerEvents && implementsPointerEvents ) {
      // accepts pointer events corresponding to the spec at http://www.w3.org/TR/pointerevents/
      input.addListener( 'pointerdown', function( domEvent ) {
        input.pointerDown( domEvent.pointerId, domEvent.pointerType, pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'pointerup', function( domEvent ) {
        input.pointerUp( domEvent.pointerId, domEvent.pointerType, pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'pointermove', function( domEvent ) {
        input.pointerMove( domEvent.pointerId, domEvent.pointerType, pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'pointerover', function( domEvent ) {
        input.pointerOver( domEvent.pointerId, domEvent.pointerType, pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'pointerout', function( domEvent ) {
        input.pointerOut( domEvent.pointerId, domEvent.pointerType, pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'pointercancel', function( domEvent ) {
        input.pointerCancel( domEvent.pointerId, domEvent.pointerType, pointFromEvent( domEvent ), domEvent );
      } );
    } else if ( this.enablePointerEvents && implementsMSPointerEvents ) {
      input.addListener( 'MSPointerDown', function( domEvent ) {
        input.pointerDown( domEvent.pointerId, msPointerType( domEvent ), pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'MSPointerUp', function( domEvent ) {
        input.pointerUp( domEvent.pointerId, msPointerType( domEvent ), pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'MSPointerMove', function( domEvent ) {
        input.pointerMove( domEvent.pointerId, msPointerType( domEvent ), pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'MSPointerOver', function( domEvent ) {
        input.pointerOver( domEvent.pointerId, msPointerType( domEvent ), pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'MSPointerOut', function( domEvent ) {
        input.pointerOut( domEvent.pointerId, msPointerType( domEvent ), pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'MSPointerCancel', function( domEvent ) {
        input.pointerCancel( domEvent.pointerId, msPointerType( domEvent ), pointFromEvent( domEvent ), domEvent );
      } );
    } else {
      input.addListener( 'mousedown', function( domEvent ) {
        input.mouseDown( pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'mouseup', function( domEvent ) {
        input.mouseUp( pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'mousemove', function( domEvent ) {
        input.mouseMove( pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'mouseover', function( domEvent ) {
        input.mouseOver( pointFromEvent( domEvent ), domEvent );
      } );
      input.addListener( 'mouseout', function( domEvent ) {
        input.mouseOut( pointFromEvent( domEvent ), domEvent );
      } );
      
      input.addListener( 'touchstart', function( domEvent ) {
        forEachChangedTouch( domEvent, function( id, point ) {
          input.touchStart( id, point, domEvent );
        } );
      } );
      input.addListener( 'touchend', function( domEvent ) {
        forEachChangedTouch( domEvent, function( id, point ) {
          input.touchEnd( id, point, domEvent );
        } );
      } );
      input.addListener( 'touchmove', function( domEvent ) {
        forEachChangedTouch( domEvent, function( id, point ) {
          input.touchMove( id, point, domEvent );
        } );
      } );
      input.addListener( 'touchcancel', function( domEvent ) {
        forEachChangedTouch( domEvent, function( id, point ) {
          input.touchCancel( id, point, domEvent );
        } );
      } );
    }
  };
  
  Scene.prototype.resizeOnWindowResize = function() {
    var scene = this;
    
    var resizer = function () {
      scene.resize( window.innerWidth, window.innerHeight );
    };
    $( window ).resize( resizer );
    resizer();
  };
  
  // in-depth check to make sure everything is layered properly
  Scene.prototype.layerAudit = function() {
    var scene = this;
    
    var boundaries = this.calculateBoundaries( null, null, null );
    assert && assert( boundaries.length === this.layers.length + 1, 'boundary count (' + boundaries.length + ') does not match layer count (' + this.layers.length + ') + 1' );
    
    // count how many 'self' trails there are
    var eachTrailUnderPaintedCount = 0;
    new scenery.Trail( this ).eachTrailUnder( function( trail ) {
      if ( trail.isPainted() ) {
        eachTrailUnderPaintedCount++;
      }
    } );
    
    var layerPaintedCount = 0;
    _.each( this.layers, function( layer ) {
      layerPaintedCount += layer.getLayerTrails().length;
    } );
    
    var layerIterationPaintedCount = 0;
    _.each( this.layers, function( layer ) {
      var selfCount = 0;
      scenery.Trail.eachPaintedTrailbetween( layer.startPaintedTrail, layer.endPaintedTrail, function( trail ) {
        selfCount++;
      }, false, scene );
      assert && assert( selfCount > 0, 'every layer must have at least one self trail' );
      layerIterationPaintedCount += selfCount;
    } );
    
    assert && assert( eachTrailUnderPaintedCount === layerPaintedCount, 'cross-referencing self trail counts: layerPaintedCount, ' + eachTrailUnderPaintedCount + ' vs ' + layerPaintedCount );
    assert && assert( eachTrailUnderPaintedCount === layerIterationPaintedCount, 'cross-referencing self trail counts: layerIterationPaintedCount, ' + eachTrailUnderPaintedCount + ' vs ' + layerIterationPaintedCount );
    
    _.each( this.layers, function( layer ) {
      var startTrail = layer.startPaintedTrail;
      var endTrail = layer.endPaintedTrail;
      
      assert && assert( startTrail.compare( endTrail ) <= 0, 'proper ordering on layer trails' );
    } );
    
    for ( var i = 1; i < this.layers.length; i++ ) {
      assert && assert( this.layers[0].startPaintedTrail.compare( this.layers[1].startPaintedTrail ) === -1, 'proper ordering of layers in scene.layers array' );
    }
    
    return true; // so we can assert( layerAudit() )
  };
  
  Scene.prototype.getDebugHTML = function() {
    var startPointer = new scenery.TrailPointer( new scenery.Trail( this ), true );
    var endPointer = new scenery.TrailPointer( new scenery.Trail( this ), false );
    
    function str( ob ) {
      return ob ? ob.toString() : ob;
    }
    
    var depth = 0;
    
    var result = '';
    
    var layerEntries = [];
    _.each( this.layers, function( layer ) {
      layer.startPointer && layer.startPointer.trail && layer.startPointer.trail.reindex();
      layer.endPointer && layer.endPointer.trail && layer.endPointer.trail.reindex();
      var startIdx = str( layer.startPointer );
      var endIndex = str( layer.endPointer );
      if ( !layerEntries[startIdx] ) {
        layerEntries[startIdx] = '';
      }
      if ( !layerEntries[endIndex] ) {
        layerEntries[endIndex] = '';
      }
      layer.startPaintedTrail.reindex();
      layer.endPaintedTrail.reindex();
      var layerInfo = layer.getId() + ' <strong>' + layer.type.name + '</strong>' +
                      ' trails: ' + ( layer.startPaintedTrail ? str( layer.startPaintedTrail ) : layer.startPaintedTrail ) +
                      ',' + ( layer.endPaintedTrail ? str( layer.endPaintedTrail ) : layer.endPaintedTrail ) +
                      ' pointers: ' + str( layer.startPointer ) +
                      ',' + str( layer.endPointer );
      layerEntries[startIdx] += '<div style="color: #080">+Layer ' + layerInfo + '</div>';
      layerEntries[endIndex] += '<div style="color: #800">-Layer ' + layerInfo + '</div>';
    } );
    
    startPointer.depthFirstUntil( endPointer, function( pointer ) {
      var div;
      var ptr = str( pointer );
      var node = pointer.trail.lastNode();
      
      function addQualifier( text ) {
          div += ' <span style="color: #008">' + text + '</span>';
        }
      
      if ( layerEntries[ptr] ) {
        result += layerEntries[ptr];
      }
      if ( pointer.isBefore ) {
        div = '<div style="margin-left: ' + ( depth * 20 ) + 'px">';
        if ( node.constructor.name ) {
          div += ' ' + node.constructor.name; // see http://stackoverflow.com/questions/332422/how-do-i-get-the-name-of-an-objects-type-in-javascript
        }
        div += ' <span style="font-weight: ' + ( node.isPainted() ? 'bold' : 'normal' ) + '">' + pointer.trail.lastNode().getId() + '</span>';
        div += ' <span style="color: #888">' + str( pointer.trail ) + '</span>';
        if ( !node._visible ) {
          addQualifier( 'invisible' );
        }
        if ( !node._pickable ) {
          addQualifier( 'unpickable' );
        }
        if ( node._clipShape ) {
          addQualifier( 'clipShape' );
        }
        if ( node._renderer ) {
          addQualifier( 'renderer:' + node._renderer.name );
        }
        if ( node._rendererOptions ) {
          addQualifier( 'rendererOptions:' + _.each( node._rendererOptions, function( option, key ) { return key + ':' + str( option ); } ).join( ',' ) );
        }
        if ( node._layerSplitBefore ) {
          addQualifier( 'layerSplitBefore' );
        }
        if ( node._layerSplitAfter ) {
          addQualifier( 'layerSplitAfter' );
        }
        if ( node._opacity < 1 ) {
          addQualifier( 'opacity:' + node._opacity );
        }
        
        var transformType = '';
        switch ( node.transform.getMatrix().type ) {
          case Matrix3.Types.IDENTITY: transformType = ''; break;
          case Matrix3.Types.TRANSLATION_2D: transformType = 'translated'; break;
          case Matrix3.Types.SCALING: transformType = 'scale'; break;
          case Matrix3.Types.AFFINE: transformType = 'affine'; break;
          case Matrix3.Types.OTHER: transformType = 'other'; break;
        }
        if ( transformType ) {
          div += ' <span style="color: #88f" title="' + node.transform.getMatrix().toString().replace( '\n', '&#10;' ) + '">' + transformType + '</span>';
        }
        div += '</div>';
        result += div;
      }
      depth += pointer.isBefore ? 1 : -1;
    }, false );
    
    return result;
  };
  
  Scene.prototype.popupDebug = function() {
    var htmlContent = '<!DOCTYPE html>'+
                      '<html lang="en">'+
                      '<head><title>Scenery Debug Snapshot</title></head>'+
                      '<body>' + this.getDebugHTML() + '</body>'+
                      '</html>';
    window.open( 'data:text/html;charset=utf-8,' + encodeURIComponent( htmlContent ) );
  };
  
  Scene.prototype.getBasicConstructor = function( propLines ) {
    return 'new scenery.Scene( $( \'#main\' ), {' + propLines + '} )';
  };
  
  Scene.prototype.toStringWithChildren = function() {
    var result = '';
    
    _.each( this._children, function( child ) {
      if ( result ) {
        result += '\n';
      }
      result += 'scene.addChild( ' + child.toString() + ' );';
    } );
    
    return result;
  };
  
  function applyCSSHacks( $main, options ) {
    // to use CSS3 transforms for performance, hide anything outside our bounds by default
    if ( !options.allowSceneOverflow ) {
      $main.css( 'overflow', 'hidden' );
    }
    
    // forward all pointer events
    $main.css( '-ms-touch-action', 'none' );
    
    if ( options.allowCSSHacks ) {
      // some css hacks (inspired from https://github.com/EightMedia/hammer.js/blob/master/hammer.js)
      (function() {
        var prefixes = [ '-webkit-', '-moz-', '-ms-', '-o-', '' ];
        var properties = {
          userSelect: 'none',
          touchCallout: 'none',
          touchAction: 'none',
          userDrag: 'none',
          tapHighlightColor: 'rgba(0,0,0,0)'
        };
        
        _.each( prefixes, function( prefix ) {
          _.each( properties, function( propertyValue, propertyName ) {
            $main.css( prefix + propertyName, propertyValue );
          } );
        } );
      })();
    }
  }
  
  return Scene;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Module that includes all Scenery dependencies, so that requiring this module will return an object
 * that consists of the entire exported 'scenery' namespace API.
 *
 * The API is actually generated by the 'scenery' module, so if this module (or all other modules) are
 * not included, the 'scenery' namespace may not be complete.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('main', [
    'SCENERY/scenery',
    'SCENERY/debug/DebugContext',
    
    'SCENERY/input/Event',
    'SCENERY/input/Input',
    'SCENERY/input/Key',
    'SCENERY/input/Mouse',
    'SCENERY/input/Pen',
    'SCENERY/input/Pointer',
    'SCENERY/input/SimpleDragHandler',
    'SCENERY/input/Touch',
    
    'SCENERY/layers/CanvasLayer',
    'SCENERY/layers/DOMLayer',
    'SCENERY/layers/Layer',
    'SCENERY/layers/LayerBoundary',
    'SCENERY/layers/LayerBuilder',
    'SCENERY/layers/LayerStrategy',
    'SCENERY/layers/LayerType',
    'SCENERY/layers/Renderer',
    'SCENERY/layers/SVGLayer',
    
    'SCENERY/nodes/Circle',
    'SCENERY/nodes/DOM',
    'SCENERY/nodes/Fillable',
    'SCENERY/nodes/HBox',
    'SCENERY/nodes/Image',
    'SCENERY/nodes/Node',
    'SCENERY/nodes/Path',
    'SCENERY/nodes/Rectangle',
    'SCENERY/nodes/Strokable',
    'SCENERY/nodes/Text',
    'SCENERY/nodes/VBox',
    
    'SCENERY/util/CanvasContextWrapper',
    'SCENERY/util/Color',
    'SCENERY/util/Font',
    'SCENERY/util/LinearGradient',
    'SCENERY/util/Pattern',
    'SCENERY/util/RadialGradient',
    'SCENERY/util/SceneImage',
    'SCENERY/util/Trail',
    'SCENERY/util/TrailInterval',
    'SCENERY/util/TrailPointer',
    'SCENERY/util/Util',
    
    'SCENERY/Scene'
  ], function(
    scenery // note: we don't need any of the other parts, we just need to specify them as dependencies so they fill in the scenery namespace
  ) {
  
  return scenery;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Module that includes all Kite dependencies, so that requiring this module will return an object
 * that consists of the entire exported 'kite' namespace API.
 *
 * The API is actually generated by the 'kite' module, so if this module (or all other modules) are
 * not included, the 'kite' namespace may not be complete.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('KITE/main', [
    'KITE/kite',
    
    'KITE/Shape',
    'KITE/pieces/Arc',
    'KITE/pieces/Close',
    'KITE/pieces/CubicCurveTo',
    'KITE/pieces/EllipticalArc',
    'KITE/pieces/LineTo',
    'KITE/pieces/MoveTo',
    'KITE/pieces/Piece',
    'KITE/pieces/QuadraticCurveTo',
    'KITE/pieces/Rect',
    'KITE/segments/Arc',
    'KITE/segments/Cubic',
    'KITE/segments/EllipticalArc',
    'KITE/segments/Line',
    'KITE/segments/Quadratic',
    'KITE/segments/Segment',
    'KITE/util/LineStyles',
    'KITE/util/Subpath'
  ], function(
    kite // note: we don't need any of the other parts, we just need to specify them as dependencies so they fill in the kite namespace
  ) {
  
  return kite;
} );

// Copyright 2002-2012, University of Colorado

/**
 * 2D convex hulls
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/ConvexHull2',['require','ASSERT/assert','DOT/dot'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'dot' );
  
  var dot = require( 'DOT/dot' );
  
  // counter-clockwise turn if > 0, clockwise turn if < 0, collinear if === 0.
  function ccw( p1, p2, p3 ) {
    return p2.minus( p1 ).crossScalar( p3.minus( p1 ) );
  }
  
  dot.ConvexHull2 = {
    // test: all collinear, multiple ways of having same angle, etc.
    
    // points is an array of Vector2 instances. see http://en.wikipedia.org/wiki/Graham_scan
    grahamScan: function( points, includeCollinear ) {
      if ( points.length <= 2 ) {
        return points;
      }
      
      // find the point 'p' with the lowest y value
      var minY = Number.POSITIVE_INFINITY;
      var p = null;
      _.each( points, function( point ) {
        if ( point.y <= minY ) {
          // if two points have the same y value, take the one with the lowest x
          if ( point.y === minY && p ) {
            if ( point.x < p.x ) {
              p = point;
            }
          } else {
            minY = point.y;
            p = point;
          }
        }
      } );
      
      // sorts the points by their angle. Between 0 and PI
      points = _.sortBy( points, function( point ) {
        return point.minus( p ).angle();
      } );
      
      // remove p from points (relies on the above statement making a defensive copy)
      points.splice( _.indexOf( points, p ), 1 );
      
      // our result array
      var result = [p];
      
      _.each( points, function( point ) {
        // ignore points equal to our starting point
        if ( p.x === point.x && p.y === point.y ) { return; }
        
        function isRightTurn() {
          if ( result.length < 2 ) {
            return false;
          }
          var cross = ccw( result[result.length-2], result[result.length-1], point );
          return includeCollinear ? ( cross < 0 ) : ( cross <= 0 );
        }
        
        while ( isRightTurn() ) {
          result.pop();
        }
        result.push( point );
      } );
      
      return result;
    }
  };
  
  return dot.ConvexHull2;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Basic width and height
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Dimension2',['require','DOT/dot'], function( require ) {
  
  
  var dot = require( 'DOT/dot' );
  
  dot.Dimension2 = function( width, height ) {
    this.width = width;
    this.height = height;
  };
  var Dimension2 = dot.Dimension2;

  Dimension2.prototype = {
    constructor: Dimension2,

    toString: function() {
      return "[" + this.width + "w, " + this.height + "h]";
    },

    equals: function( other ) {
      return this.width === other.width && this.height === other.height;
    }
  };
  
  return Dimension2;
} );

// Copyright 2002-2012, University of Colorado

/**
 * LU decomposition, based on Jama
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/LUDecomposition',['require','DOT/dot'], function( require ) {
  
  
  var dot = require( 'DOT/dot' );
  
  var Float32Array = window.Float32Array || Array;
  
  // require( 'DOT/Matrix' ); // commented out so Require.js doesn't complain about the circular dependency

  dot.LUDecomposition = function( matrix ) {
    var i, j, k;

    this.matrix = matrix;

    // TODO: size!
    this.LU = matrix.getArrayCopy();
    var LU = this.LU;
    this.m = matrix.getRowDimension();
    var m = this.m;
    this.n = matrix.getColumnDimension();
    var n = this.n;
    this.piv = new Uint32Array( m );
    for ( i = 0; i < m; i++ ) {
      this.piv[i] = i;
    }
    this.pivsign = 1;
    var LUcolj = new Float32Array( m );

    // Outer loop.

    for ( j = 0; j < n; j++ ) {

      // Make a copy of the j-th column to localize references.
      for ( i = 0; i < m; i++ ) {
        LUcolj[i] = LU[matrix.index( i, j )];
      }

      // Apply previous transformations.

      for ( i = 0; i < m; i++ ) {
        // Most of the time is spent in the following dot product.
        var kmax = Math.min( i, j );
        var s = 0.0;
        for ( k = 0; k < kmax; k++ ) {
          var ik = matrix.index( i, k );
          s += LU[ik] * LUcolj[k];
        }

        LUcolj[i] -= s;
        LU[matrix.index( i, j )] = LUcolj[i];
      }

      // Find pivot and exchange if necessary.

      var p = j;
      for ( i = j + 1; i < m; i++ ) {
        if ( Math.abs( LUcolj[i] ) > Math.abs( LUcolj[p] ) ) {
          p = i;
        }
      }
      if ( p !== j ) {
        for ( k = 0; k < n; k++ ) {
          var pk = matrix.index( p, k );
          var jk = matrix.index( j, k );
          var t = LU[pk];
          LU[pk] = LU[jk];
          LU[jk] = t;
        }
        k = this.piv[p];
        this.piv[p] = this.piv[j];
        this.piv[j] = k;
        this.pivsign = -this.pivsign;
      }

      // Compute multipliers.

      if ( j < m && LU[this.matrix.index( j, j )] !== 0.0 ) {
        for ( i = j + 1; i < m; i++ ) {
          LU[matrix.index( i, j )] /= LU[matrix.index( j, j )];
        }
      }
    }
  };
  var LUDecomposition = dot.LUDecomposition;

  LUDecomposition.prototype = {
    constructor: LUDecomposition,

    isNonsingular: function() {
      for ( var j = 0; j < this.n; j++ ) {
        var index = this.matrix.index( j, j );
        if ( this.LU[index] === 0 ) {
          return false;
        }
      }
      return true;
    },

    getL: function() {
      var result = new dot.Matrix( this.m, this.n );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          if ( i > j ) {
            result.entries[result.index( i, j )] = this.LU[this.matrix.index( i, j )];
          }
          else if ( i === j ) {
            result.entries[result.index( i, j )] = 1.0;
          }
          else {
            result.entries[result.index( i, j )] = 0.0;
          }
        }
      }
      return result;
    },

    getU: function() {
      var result = new dot.Matrix( this.n, this.n );
      for ( var i = 0; i < this.n; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          if ( i <= j ) {
            result.entries[result.index( i, j )] = this.LU[this.matrix.index( i, j )];
          }
          else {
            result.entries[result.index( i, j )] = 0.0;
          }
        }
      }
      return result;
    },

    getPivot: function() {
      var p = new Uint32Array( this.m );
      for ( var i = 0; i < this.m; i++ ) {
        p[i] = this.piv[i];
      }
      return p;
    },

    getDoublePivot: function() {
      var vals = new Float32Array( this.m );
      for ( var i = 0; i < this.m; i++ ) {
        vals[i] = this.piv[i];
      }
      return vals;
    },

    det: function() {
      if ( this.m !== this.n ) {
        throw new Error( "Matrix must be square." );
      }
      var d = this.pivsign;
      for ( var j = 0; j < this.n; j++ ) {
        d *= this.LU[this.matrix.index( j, j )];
      }
      return d;
    },

    solve: function( matrix ) {
      var i, j, k;
      if ( matrix.getRowDimension() !== this.m ) {
        throw new Error( "Matrix row dimensions must agree." );
      }
      if ( !this.isNonsingular() ) {
        throw new Error( "Matrix is singular." );
      }

      // Copy right hand side with pivoting
      var nx = matrix.getColumnDimension();
      var Xmat = matrix.getArrayRowMatrix( this.piv, 0, nx - 1 );

      // Solve L*Y = B(piv,:)
      for ( k = 0; k < this.n; k++ ) {
        for ( i = k + 1; i < this.n; i++ ) {
          for ( j = 0; j < nx; j++ ) {
            Xmat.entries[Xmat.index( i, j )] -= Xmat.entries[Xmat.index( k, j )] * this.LU[this.matrix.index( i, k )];
          }
        }
      }

      // Solve U*X = Y;
      for ( k = this.n - 1; k >= 0; k-- ) {
        for ( j = 0; j < nx; j++ ) {
          Xmat.entries[Xmat.index( k, j )] /= this.LU[this.matrix.index( k, k )];
        }
        for ( i = 0; i < k; i++ ) {
          for ( j = 0; j < nx; j++ ) {
            Xmat.entries[Xmat.index( i, j )] -= Xmat.entries[Xmat.index( k, j )] * this.LU[this.matrix.index( i, k )];
          }
        }
      }
      return Xmat;
    }
  };
  
  return LUDecomposition;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Tests whether a reference is to an array.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('PHET_CORE/isArray',['require'], function( require ) {
  
  
  return function isArray( array ) {
    // yes, this is actually how to do this. see http://stackoverflow.com/questions/4775722/javascript-check-if-object-is-array
    return Object.prototype.toString.call( array ) === '[object Array]';
  };
} );

// Copyright 2002-2012, University of Colorado

/**
 * SVD decomposition, based on Jama
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/SingularValueDecomposition',['require','DOT/dot'], function( require ) {
  
  
  var dot = require( 'DOT/dot' );
  
  var Float32Array = window.Float32Array || Array;
  
  // require( 'DOT/Matrix' ); // commented out so Require.js doesn't complain about the circular dependency

  dot.SingularValueDecomposition = function( matrix ) {
    this.matrix = matrix;

    var Arg = matrix;

    // Derived from LINPACK code.
    // Initialize.
    var A = Arg.getArrayCopy();
    this.m = Arg.getRowDimension();
    this.n = Arg.getColumnDimension();
    var m = this.m;
    var n = this.n;

    var min = Math.min;
    var max = Math.max;
    var pow = Math.pow;
    var abs = Math.abs;

    /* Apparently the failing cases are only a proper subset of (m<n),
     so let's not throw error.  Correct fix to come later?
     if (m<n) {
     throw new IllegalArgumentException("Jama SVD only works for m >= n"); }
     */
    var nu = min( m, n );
    this.s = new Float32Array( min( m + 1, n ) );
    var s = this.s;
    this.U = new Float32Array( m * nu );
    var U = this.U;
    this.V = new Float32Array( n * n );
    var V = this.V;
    var e = new Float32Array( n );
    var work = new Float32Array( m );
    var wantu = true;
    var wantv = true;

    var i, j, k, t, f;
    var cs,sn;

    var hypot = dot.Matrix.hypot;

    // Reduce A to bidiagonal form, storing the diagonal elements
    // in s and the super-diagonal elements in e.

    var nct = min( m - 1, n );
    var nrt = max( 0, min( n - 2, m ) );
    for ( k = 0; k < max( nct, nrt ); k++ ) {
      if ( k < nct ) {

        // Compute the transformation for the k-th column and
        // place the k-th diagonal in s[k].
        // Compute 2-norm of k-th column without under/overflow.
        s[k] = 0;
        for ( i = k; i < m; i++ ) {
          s[k] = hypot( s[k], A[i * n + k] );
        }
        if ( s[k] !== 0.0 ) {
          if ( A[k * n + k] < 0.0 ) {
            s[k] = -s[k];
          }
          for ( i = k; i < m; i++ ) {
            A[i * n + k] /= s[k];
          }
          A[k * n + k] += 1.0;
        }
        s[k] = -s[k];
      }
      for ( j = k + 1; j < n; j++ ) {
        if ( (k < nct) && (s[k] !== 0.0) ) {

          // Apply the transformation.

          t = 0;
          for ( i = k; i < m; i++ ) {
            t += A[i * n + k] * A[i * n + j];
          }
          t = -t / A[k * n + k];
          for ( i = k; i < m; i++ ) {
            A[i * n + j] += t * A[i * n + k];
          }
        }

        // Place the k-th row of A into e for the
        // subsequent calculation of the row transformation.

        e[j] = A[k * n + j];
      }
      if ( wantu && (k < nct) ) {

        // Place the transformation in U for subsequent back
        // multiplication.

        for ( i = k; i < m; i++ ) {
          U[i * nu + k] = A[i * n + k];
        }
      }
      if ( k < nrt ) {

        // Compute the k-th row transformation and place the
        // k-th super-diagonal in e[k].
        // Compute 2-norm without under/overflow.
        e[k] = 0;
        for ( i = k + 1; i < n; i++ ) {
          e[k] = hypot( e[k], e[i] );
        }
        if ( e[k] !== 0.0 ) {
          if ( e[k + 1] < 0.0 ) {
            e[k] = -e[k];
          }
          for ( i = k + 1; i < n; i++ ) {
            e[i] /= e[k];
          }
          e[k + 1] += 1.0;
        }
        e[k] = -e[k];
        if ( (k + 1 < m) && (e[k] !== 0.0) ) {

          // Apply the transformation.

          for ( i = k + 1; i < m; i++ ) {
            work[i] = 0.0;
          }
          for ( j = k + 1; j < n; j++ ) {
            for ( i = k + 1; i < m; i++ ) {
              work[i] += e[j] * A[i * n + j];
            }
          }
          for ( j = k + 1; j < n; j++ ) {
            t = -e[j] / e[k + 1];
            for ( i = k + 1; i < m; i++ ) {
              A[i * n + j] += t * work[i];
            }
          }
        }
        if ( wantv ) {

          // Place the transformation in V for subsequent
          // back multiplication.

          for ( i = k + 1; i < n; i++ ) {
            V[i * n + k] = e[i];
          }
        }
      }
    }

    // Set up the final bidiagonal matrix or order p.

    var p = min( n, m + 1 );
    if ( nct < n ) {
      s[nct] = A[nct * n + nct];
    }
    if ( m < p ) {
      s[p - 1] = 0.0;
    }
    if ( nrt + 1 < p ) {
      e[nrt] = A[nrt * n + p - 1];
    }
    e[p - 1] = 0.0;

    // If required, generate U.

    if ( wantu ) {
      for ( j = nct; j < nu; j++ ) {
        for ( i = 0; i < m; i++ ) {
          U[i * nu + j] = 0.0;
        }
        U[j * nu + j] = 1.0;
      }
      for ( k = nct - 1; k >= 0; k-- ) {
        if ( s[k] !== 0.0 ) {
          for ( j = k + 1; j < nu; j++ ) {
            t = 0;
            for ( i = k; i < m; i++ ) {
              t += U[i * nu + k] * U[i * nu + j];
            }
            t = -t / U[k * nu + k];
            for ( i = k; i < m; i++ ) {
              U[i * nu + j] += t * U[i * nu + k];
            }
          }
          for ( i = k; i < m; i++ ) {
            U[i * nu + k] = -U[i * nu + k];
          }
          U[k * nu + k] = 1.0 + U[k * nu + k];
          for ( i = 0; i < k - 1; i++ ) {
            U[i * nu + k] = 0.0;
          }
        }
        else {
          for ( i = 0; i < m; i++ ) {
            U[i * nu + k] = 0.0;
          }
          U[k * nu + k] = 1.0;
        }
      }
    }

    // If required, generate V.

    if ( wantv ) {
      for ( k = n - 1; k >= 0; k-- ) {
        if ( (k < nrt) && (e[k] !== 0.0) ) {
          for ( j = k + 1; j < nu; j++ ) {
            t = 0;
            for ( i = k + 1; i < n; i++ ) {
              t += V[i * n + k] * V[i * n + j];
            }
            t = -t / V[k + 1 * n + k];
            for ( i = k + 1; i < n; i++ ) {
              V[i * n + j] += t * V[i * n + k];
            }
          }
        }
        for ( i = 0; i < n; i++ ) {
          V[i * n + k] = 0.0;
        }
        V[k * n + k] = 1.0;
      }
    }

    // Main iteration loop for the singular values.

    var pp = p - 1;
    var iter = 0;
    var eps = pow( 2.0, -52.0 );
    var tiny = pow( 2.0, -966.0 );
    while ( p > 0 ) {
      var kase;

      // Here is where a test for too many iterations would go.

      // This section of the program inspects for
      // negligible elements in the s and e arrays.  On
      // completion the variables kase and k are set as follows.

      // kase = 1   if s(p) and e[k-1] are negligible and k<p
      // kase = 2   if s(k) is negligible and k<p
      // kase = 3   if e[k-1] is negligible, k<p, and
      //        s(k), ..., s(p) are not negligible (qr step).
      // kase = 4   if e(p-1) is negligible (convergence).

      for ( k = p - 2; k >= -1; k-- ) {
        if ( k === -1 ) {
          break;
        }
        if ( abs( e[k] ) <=
           tiny + eps * (abs( s[k] ) + abs( s[k + 1] )) ) {
          e[k] = 0.0;
          break;
        }
      }
      if ( k === p - 2 ) {
        kase = 4;
      }
      else {
        var ks;
        for ( ks = p - 1; ks >= k; ks-- ) {
          if ( ks === k ) {
            break;
          }
          t = (ks !== p ? abs( e[ks] ) : 0) +
            (ks !== k + 1 ? abs( e[ks - 1] ) : 0);
          if ( abs( s[ks] ) <= tiny + eps * t ) {
            s[ks] = 0.0;
            break;
          }
        }
        if ( ks === k ) {
          kase = 3;
        }
        else if ( ks === p - 1 ) {
          kase = 1;
        }
        else {
          kase = 2;
          k = ks;
        }
      }
      k++;

      // Perform the task indicated by kase.

      switch( kase ) {

        // Deflate negligible s(p).

        case 1:
        {
          f = e[p - 2];
          e[p - 2] = 0.0;
          for ( j = p - 2; j >= k; j-- ) {
            t = hypot( s[j], f );
            cs = s[j] / t;
            sn = f / t;
            s[j] = t;
            if ( j !== k ) {
              f = -sn * e[j - 1];
              e[j - 1] = cs * e[j - 1];
            }
            if ( wantv ) {
              for ( i = 0; i < n; i++ ) {
                t = cs * V[i * n + j] + sn * V[i * n + p - 1];
                V[i * n + p - 1] = -sn * V[i * n + j] + cs * V[i * n + p - 1];
                V[i * n + j] = t;
              }
            }
          }
        }
          break;

        // Split at negligible s(k).

        case 2:
        {
          f = e[k - 1];
          e[k - 1] = 0.0;
          for ( j = k; j < p; j++ ) {
            t = hypot( s[j], f );
            cs = s[j] / t;
            sn = f / t;
            s[j] = t;
            f = -sn * e[j];
            e[j] = cs * e[j];
            if ( wantu ) {
              for ( i = 0; i < m; i++ ) {
                t = cs * U[i * nu + j] + sn * U[i * nu + k - 1];
                U[i * nu + k - 1] = -sn * U[i * nu + j] + cs * U[i * nu + k - 1];
                U[i * nu + j] = t;
              }
            }
          }
        }
          break;

        // Perform one qr step.

        case 3:
        {

          // Calculate the shift.

          var scale = max( max( max( max(
              abs( s[p - 1] ), abs( s[p - 2] ) ), abs( e[p - 2] ) ),
                          abs( s[k] ) ), abs( e[k] ) );
          var sp = s[p - 1] / scale;
          var spm1 = s[p - 2] / scale;
          var epm1 = e[p - 2] / scale;
          var sk = s[k] / scale;
          var ek = e[k] / scale;
          var b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
          var c = (sp * epm1) * (sp * epm1);
          var shift = 0.0;
          if ( (b !== 0.0) || (c !== 0.0) ) {
            shift = Math.sqrt( b * b + c );
            if ( b < 0.0 ) {
              shift = -shift;
            }
            shift = c / (b + shift);
          }
          f = (sk + sp) * (sk - sp) + shift;
          var g = sk * ek;

          // Chase zeros.

          for ( j = k; j < p - 1; j++ ) {
            t = hypot( f, g );
            cs = f / t;
            sn = g / t;
            if ( j !== k ) {
              e[j - 1] = t;
            }
            f = cs * s[j] + sn * e[j];
            e[j] = cs * e[j] - sn * s[j];
            g = sn * s[j + 1];
            s[j + 1] = cs * s[j + 1];
            if ( wantv ) {
              for ( i = 0; i < n; i++ ) {
                t = cs * V[i * n + j] + sn * V[i * n + j + 1];
                V[i * n + j + 1] = -sn * V[i * n + j] + cs * V[i * n + j + 1];
                V[i * n + j] = t;
              }
            }
            t = hypot( f, g );
            cs = f / t;
            sn = g / t;
            s[j] = t;
            f = cs * e[j] + sn * s[j + 1];
            s[j + 1] = -sn * e[j] + cs * s[j + 1];
            g = sn * e[j + 1];
            e[j + 1] = cs * e[j + 1];
            if ( wantu && (j < m - 1) ) {
              for ( i = 0; i < m; i++ ) {
                t = cs * U[i * nu + j] + sn * U[i * nu + j + 1];
                U[i * nu + j + 1] = -sn * U[i * nu + j] + cs * U[i * nu + j + 1];
                U[i * nu + j] = t;
              }
            }
          }
          e[p - 2] = f;
          iter = iter + 1;
        }
          break;

        // Convergence.

        case 4:
        {

          // Make the singular values positive.

          if ( s[k] <= 0.0 ) {
            s[k] = (s[k] < 0.0 ? -s[k] : 0.0);
            if ( wantv ) {
              for ( i = 0; i <= pp; i++ ) {
                V[i * n + k] = -V[i * n + k];
              }
            }
          }

          // Order the singular values.

          while ( k < pp ) {
            if ( s[k] >= s[k + 1] ) {
              break;
            }
            t = s[k];
            s[k] = s[k + 1];
            s[k + 1] = t;
            if ( wantv && (k < n - 1) ) {
              for ( i = 0; i < n; i++ ) {
                t = V[i * n + k + 1];
                V[i * n + k + 1] = V[i * n + k];
                V[i * n + k] = t;
              }
            }
            if ( wantu && (k < m - 1) ) {
              for ( i = 0; i < m; i++ ) {
                t = U[i * nu + k + 1];
                U[i * nu + k + 1] = U[i * nu + k];
                U[i * nu + k] = t;
              }
            }
            k++;
          }
          iter = 0;
          p--;
        }
          break;
      }
    }
  };
  var SingularValueDecomposition = dot.SingularValueDecomposition;

  SingularValueDecomposition.prototype = {
    constructor: SingularValueDecomposition,

    getU: function() {
      return new dot.Matrix( this.m, Math.min( this.m + 1, this.n ), this.U, true ); // the "fast" flag added, since U is Float32Array
    },

    getV: function() {
      return new dot.Matrix( this.n, this.n, this.V, true );
    },

    getSingularValues: function() {
      return this.s;
    },

    getS: function() {
      var result = new dot.Matrix( this.n, this.n );
      for ( var i = 0; i < this.n; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          result.entries[result.index( i, j )] = 0.0;
        }
        result.entries[result.index( i, i )] = this.s[i];
      }
      return result;
    },

    norm2: function() {
      return this.s[0];
    },

    cond: function() {
      return this.s[0] / this.s[Math.min( this.m, this.n ) - 1];
    },

    rank: function() {
      // changed to 23 from 52 (bits of mantissa), since we are using floats here!
      var eps = Math.pow( 2.0, -23.0 );
      var tol = Math.max( this.m, this.n ) * this.s[0] * eps;
      var r = 0;
      for ( var i = 0; i < this.s.length; i++ ) {
        if ( this.s[i] > tol ) {
          r++;
        }
      }
      return r;
    }
  };
} );

// Copyright 2002-2012, University of Colorado

/**
 * QR decomposition, based on Jama
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/QRDecomposition',['require','DOT/dot'], function( require ) {
  
  
  var dot = require( 'DOT/dot' );
  
  var Float32Array = window.Float32Array || Array;
  
  // require( 'DOT/Matrix' ); // commented out so Require.js doesn't complain about the circular dependency

  dot.QRDecomposition = function( matrix ) {
    this.matrix = matrix;

    // TODO: size!
    this.QR = matrix.getArrayCopy();
    var QR = this.QR;
    this.m = matrix.getRowDimension();
    var m = this.m;
    this.n = matrix.getColumnDimension();
    var n = this.n;

    this.Rdiag = new Float32Array( n );

    var i, j, k;

    // Main loop.
    for ( k = 0; k < n; k++ ) {
      // Compute 2-norm of k-th column without under/overflow.
      var nrm = 0;
      for ( i = k; i < m; i++ ) {
        nrm = dot.Matrix.hypot( nrm, QR[this.matrix.index( i, k )] );
      }

      if ( nrm !== 0.0 ) {
        // Form k-th Householder vector.
        if ( QR[this.matrix.index( k, k )] < 0 ) {
          nrm = -nrm;
        }
        for ( i = k; i < m; i++ ) {
          QR[this.matrix.index( i, k )] /= nrm;
        }
        QR[this.matrix.index( k, k )] += 1.0;

        // Apply transformation to remaining columns.
        for ( j = k + 1; j < n; j++ ) {
          var s = 0.0;
          for ( i = k; i < m; i++ ) {
            s += QR[this.matrix.index( i, k )] * QR[this.matrix.index( i, j )];
          }
          s = -s / QR[this.matrix.index( k, k )];
          for ( i = k; i < m; i++ ) {
            QR[this.matrix.index( i, j )] += s * QR[this.matrix.index( i, k )];
          }
        }
      }
      this.Rdiag[k] = -nrm;
    }
  };
  var QRDecomposition = dot.QRDecomposition;

  QRDecomposition.prototype = {
    constructor: QRDecomposition,

    isFullRank: function() {
      for ( var j = 0; j < this.n; j++ ) {
        if ( this.Rdiag[j] === 0 ) {
          return false;
        }
      }
      return true;
    },

    getH: function() {
      var result = new dot.Matrix( this.m, this.n );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          if ( i >= j ) {
            result.entries[result.index( i, j )] = this.QR[this.matrix.index( i, j )];
          }
          else {
            result.entries[result.index( i, j )] = 0.0;
          }
        }
      }
      return result;
    },

    getR: function() {
      var result = new dot.Matrix( this.n, this.n );
      for ( var i = 0; i < this.n; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          if ( i < j ) {
            result.entries[result.index( i, j )] = this.QR[this.matrix.index( i, j )];
          }
          else if ( i === j ) {
            result.entries[result.index( i, j )] = this.Rdiag[i];
          }
          else {
            result.entries[result.index( i, j )] = 0.0;
          }
        }
      }
      return result;
    },

    getQ: function() {
      var i, j, k;
      var result = new dot.Matrix( this.m, this.n );
      for ( k = this.n - 1; k >= 0; k-- ) {
        for ( i = 0; i < this.m; i++ ) {
          result.entries[result.index( i, k )] = 0.0;
        }
        result.entries[result.index( k, k )] = 1.0;
        for ( j = k; j < this.n; j++ ) {
          if ( this.QR[this.matrix.index( k, k )] !== 0 ) {
            var s = 0.0;
            for ( i = k; i < this.m; i++ ) {
              s += this.QR[this.matrix.index( i, k )] * result.entries[result.index( i, j )];
            }
            s = -s / this.QR[this.matrix.index( k, k )];
            for ( i = k; i < this.m; i++ ) {
              result.entries[result.index( i, j )] += s * this.QR[this.matrix.index( i, k )];
            }
          }
        }
      }
      return result;
    },

    solve: function( matrix ) {
      if ( matrix.getRowDimension() !== this.m ) {
        throw new Error( "Matrix row dimensions must agree." );
      }
      if ( !this.isFullRank() ) {
        throw new Error( "Matrix is rank deficient." );
      }

      var i, j, k;

      // Copy right hand side
      var nx = matrix.getColumnDimension();
      var X = matrix.getArrayCopy();

      // Compute Y = transpose(Q)*matrix
      for ( k = 0; k < this.n; k++ ) {
        for ( j = 0; j < nx; j++ ) {
          var s = 0.0;
          for ( i = k; i < this.m; i++ ) {
            s += this.QR[this.matrix.index( i, k )] * X[matrix.index( i, j )];
          }
          s = -s / this.QR[this.matrix.index( k, k )];
          for ( i = k; i < this.m; i++ ) {
            X[matrix.index( i, j )] += s * this.QR[this.matrix.index( i, k )];
          }
        }
      }

      // Solve R*X = Y;
      for ( k = this.n - 1; k >= 0; k-- ) {
        for ( j = 0; j < nx; j++ ) {
          X[matrix.index( k, j )] /= this.Rdiag[k];
        }
        for ( i = 0; i < k; i++ ) {
          for ( j = 0; j < nx; j++ ) {
            X[matrix.index( i, j )] -= X[matrix.index( k, j )] * this.QR[this.matrix.index( i, k )];
          }
        }
      }
      return new dot.Matrix( X, this.n, nx ).getMatrix( 0, this.n - 1, 0, nx - 1 );
    }
  };
  
  return QRDecomposition;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Arbitrary-dimensional matrix, based on Jama
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Matrix',['require','ASSERT/assert','DOT/dot','PHET_CORE/isArray','DOT/SingularValueDecomposition','DOT/LUDecomposition','DOT/QRDecomposition','DOT/Vector2','DOT/Vector3','DOT/Vector4'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'dot' );
  
  var dot = require( 'DOT/dot' );
  
  var Float32Array = window.Float32Array || Array;
  
  var isArray = require( 'PHET_CORE/isArray' );
  
  require( 'DOT/SingularValueDecomposition' );
  require( 'DOT/LUDecomposition' );
  require( 'DOT/QRDecomposition' );
  require( 'DOT/Vector2' );
  require( 'DOT/Vector3' );
  require( 'DOT/Vector4' );
  
  dot.Matrix = function( m, n, filler, fast ) {
    this.m = m;
    this.n = n;

    var size = m * n;
    this.size = size;
    var i;

    if ( fast ) {
      this.entries = filler;
    }
    else {
      if ( !filler ) {
        filler = 0;
      }

      // entries stored in row-major format
      this.entries = new Float32Array( size );

      if ( isArray( filler ) ) {
        assert && assert( filler.length === size );

        for ( i = 0; i < size; i++ ) {
          this.entries[i] = filler[i];
        }
      }
      else {
        for ( i = 0; i < size; i++ ) {
          this.entries[i] = filler;
        }
      }
    }
  };
  var Matrix = dot.Matrix;

  /** sqrt(a^2 + b^2) without under/overflow. **/
  Matrix.hypot = function hypot( a, b ) {
    var r;
    if ( Math.abs( a ) > Math.abs( b ) ) {
      r = b / a;
      r = Math.abs( a ) * Math.sqrt( 1 + r * r );
    }
    else if ( b !== 0 ) {
      r = a / b;
      r = Math.abs( b ) * Math.sqrt( 1 + r * r );
    }
    else {
      r = 0.0;
    }
    return r;
  };

  Matrix.prototype = {
    constructor: Matrix,

    copy: function() {
      var result = new Matrix( this.m, this.n );
      for ( var i = 0; i < this.size; i++ ) {
        result.entries[i] = this.entries[i];
      }
      return result;
    },

    getArray: function() {
      return this.entries;
    },

    getArrayCopy: function() {
      return new Float32Array( this.entries );
    },

    getRowDimension: function() {
      return this.m;
    },

    getColumnDimension: function() {
      return this.n;
    },

    // TODO: inline this places if we aren't using an inlining compiler! (check performance)
    index: function( i, j ) {
      return i * this.n + j;
    },

    get: function( i, j ) {
      return this.entries[this.index( i, j )];
    },

    set: function( i, j, s ) {
      this.entries[this.index( i, j )] = s;
    },

    getMatrix: function( i0, i1, j0, j1 ) {
      var result = new Matrix( i1 - i0 + 1, j1 - j0 + 1 );
      for ( var i = i0; i <= i1; i++ ) {
        for ( var j = j0; j <= j1; j++ ) {
          result.entries[result.index( i - i0, j - j0 )] = this.entries[this.index( i, j )];
        }
      }
      return result;
    },

    // getMatrix (int[] r, int j0, int j1)
    getArrayRowMatrix: function( r, j0, j1 ) {
      var result = new Matrix( r.length, j1 - j0 + 1 );
      for ( var i = 0; i < r.length; i++ ) {
        for ( var j = j0; j <= j1; j++ ) {
          result.entries[result.index( i, j - j0 )] = this.entries[this.index( r[i], j )];
        }
      }
      return result;
    },

    transpose: function() {
      var result = new Matrix( this.n, this.m );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          result.entries[result.index( j, i )] = this.entries[this.index( i, j )];
        }
      }
      return result;
    },

    norm1: function() {
      var f = 0;
      for ( var j = 0; j < this.n; j++ ) {
        var s = 0;
        for ( var i = 0; i < this.m; i++ ) {
          s += Math.abs( this.entries[ this.index( i, j ) ] );
        }
        f = Math.max( f, s );
      }
      return f;
    },

    norm2: function() {
      return (new dot.SingularValueDecomposition( this ).norm2());
    },

    normInf: function() {
      var f = 0;
      for ( var i = 0; i < this.m; i++ ) {
        var s = 0;
        for ( var j = 0; j < this.n; j++ ) {
          s += Math.abs( this.entries[ this.index( i, j ) ] );
        }
        f = Math.max( f, s );
      }
      return f;
    },

    normF: function() {
      var f = 0;
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          f = Matrix.hypot( f, this.entries[ this.index( i, j ) ] );
        }
      }
      return f;
    },

    uminus: function() {
      var result = new Matrix( this.m, this.n );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          result.entries[result.index( i, j )] = -this.entries[ this.index( i, j ) ];
        }
      }
      return result;
    },

    plus: function( matrix ) {
      this.checkMatrixDimensions( matrix );
      var result = new Matrix( this.m, this.n );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          var index = result.index( i, j );
          result.entries[index] = this.entries[index] + matrix.entries[index];
        }
      }
      return result;
    },

    plusEquals: function( matrix ) {
      this.checkMatrixDimensions( matrix );
      var result = new Matrix( this.m, this.n );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          var index = result.index( i, j );
          this.entries[index] = this.entries[index] + matrix.entries[index];
        }
      }
      return this;
    },

    minus: function( matrix ) {
      this.checkMatrixDimensions( matrix );
      var result = new Matrix( this.m, this.n );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          var index = this.index( i, j );
          result.entries[index] = this.entries[index] - matrix.entries[index];
        }
      }
      return result;
    },

    minusEquals: function( matrix ) {
      this.checkMatrixDimensions( matrix );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          var index = this.index( i, j );
          this.entries[index] = this.entries[index] - matrix.entries[index];
        }
      }
      return this;
    },

    arrayTimes: function( matrix ) {
      this.checkMatrixDimensions( matrix );
      var result = new Matrix( this.m, this.n );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          var index = result.index( i, j );
          result.entries[index] = this.entries[index] * matrix.entries[index];
        }
      }
      return result;
    },

    arrayTimesEquals: function( matrix ) {
      this.checkMatrixDimensions( matrix );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          var index = this.index( i, j );
          this.entries[index] = this.entries[index] * matrix.entries[index];
        }
      }
      return this;
    },

    arrayRightDivide: function( matrix ) {
      this.checkMatrixDimensions( matrix );
      var result = new Matrix( this.m, this.n );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          var index = this.index( i, j );
          result.entries[index] = this.entries[index] / matrix.entries[index];
        }
      }
      return result;
    },

    arrayRightDivideEquals: function( matrix ) {
      this.checkMatrixDimensions( matrix );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          var index = this.index( i, j );
          this.entries[index] = this.entries[index] / matrix.entries[index];
        }
      }
      return this;
    },

    arrayLeftDivide: function( matrix ) {
      this.checkMatrixDimensions( matrix );
      var result = new Matrix( this.m, this.n );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          var index = this.index( i, j );
          result.entries[index] = matrix.entries[index] / this.entries[index];
        }
      }
      return result;
    },

    arrayLeftDivideEquals: function( matrix ) {
      this.checkMatrixDimensions( matrix );
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          var index = this.index( i, j );
          this.entries[index] = matrix.entries[index] / this.entries[index];
        }
      }
      return this;
    },

    times: function( matrixOrScalar ) {
      var result;
      var i, j, k, s;
      var matrix;
      if ( matrixOrScalar.isMatrix ) {
        matrix = matrixOrScalar;
        if ( matrix.m !== this.n ) {
          throw new Error( "Matrix inner dimensions must agree." );
        }
        result = new Matrix( this.m, matrix.n );
        var matrixcolj = new Float32Array( this.n );
        for ( j = 0; j < matrix.n; j++ ) {
          for ( k = 0; k < this.n; k++ ) {
            matrixcolj[k] = matrix.entries[ matrix.index( k, j ) ];
          }
          for ( i = 0; i < this.m; i++ ) {
            s = 0;
            for ( k = 0; k < this.n; k++ ) {
              s += this.entries[this.index( i, k )] * matrixcolj[k];
            }
            result.entries[result.index( i, j )] = s;
          }
        }
        return result;
      }
      else {
        s = matrixOrScalar;
        result = new Matrix( this.m, this.n );
        for ( i = 0; i < this.m; i++ ) {
          for ( j = 0; j < this.n; j++ ) {
            result.entries[result.index( i, j )] = s * this.entries[this.index( i, j )];
          }
        }
        return result;
      }
    },

    timesEquals: function( s ) {
      for ( var i = 0; i < this.m; i++ ) {
        for ( var j = 0; j < this.n; j++ ) {
          var index = this.index( i, j );
          this.entries[index] = s * this.entries[index];
        }
      }
      return this;
    },

    solve: function( matrix ) {
      return (this.m === this.n ? (new dot.LUDecomposition( this )).solve( matrix ) :
          (new dot.QRDecomposition( this )).solve( matrix ));
    },

    solveTranspose: function( matrix ) {
      return this.transpose().solve( matrix.transpose() );
    },

    inverse: function() {
      return this.solve( Matrix.identity( this.m, this.m ) );
    },

    det: function() {
      return new dot.LUDecomposition( this ).det();
    },

    rank: function() {
      return new dot.SingularValueDecomposition( this ).rank();
    },

    cond: function() {
      return new dot.SingularValueDecomposition( this ).cond();
    },

    trace: function() {
      var t = 0;
      for ( var i = 0; i < Math.min( this.m, this.n ); i++ ) {
        t += this.entries[ this.index( i, i ) ];
      }
      return t;
    },

    checkMatrixDimensions: function( matrix ) {
      if ( matrix.m !== this.m || matrix.n !== this.n ) {
        throw new Error( "Matrix dimensions must agree." );
      }
    },

    toString: function() {
      var result = "";
      result += "dim: " + this.getRowDimension() + "x" + this.getColumnDimension() + "\n";
      for ( var row = 0; row < this.getRowDimension(); row++ ) {
        for ( var col = 0; col < this.getColumnDimension(); col++ ) {
          result += this.get( row, col ) + " ";
        }
        result += "\n";
      }
      return result;
    },

    // returns a vector that is contained in the specified column
    extractVector2: function( column ) {
      assert && assert( this.m === 2 ); // rows should match vector dimension
      return new dot.Vector2( this.get( 0, column ), this.get( 1, column ) );
    },

    // returns a vector that is contained in the specified column
    extractVector3: function( column ) {
      assert && assert( this.m === 3 ); // rows should match vector dimension
      return new dot.Vector3( this.get( 0, column ), this.get( 1, column ), this.get( 2, column ) );
    },

    // returns a vector that is contained in the specified column
    extractVector4: function( column ) {
      assert && assert( this.m === 4 ); // rows should match vector dimension
      return new dot.Vector4( this.get( 0, column ), this.get( 1, column ), this.get( 2, column ), this.get( 3, column ) );
    },

    isMatrix: true
  };

  Matrix.identity = function( m, n ) {
    var result = new Matrix( m, n );
    for ( var i = 0; i < m; i++ ) {
      for ( var j = 0; j < n; j++ ) {
        result.entries[result.index( i, j )] = (i === j ? 1.0 : 0.0);
      }
    }
    return result;
  };

  Matrix.rowVector2 = function( vector ) {
    return new Matrix( 1, 2, [vector.x, vector.y] );
  };

  Matrix.rowVector3 = function( vector ) {
    return new Matrix( 1, 3, [vector.x, vector.y, vector.z] );
  };

  Matrix.rowVector4 = function( vector ) {
    return new Matrix( 1, 4, [vector.x, vector.y, vector.z, vector.w] );
  };

  Matrix.rowVector = function( vector ) {
    if ( vector.isVector2 ) {
      return Matrix.rowVector2( vector );
    }
    else if ( vector.isVector3 ) {
      return Matrix.rowVector3( vector );
    }
    else if ( vector.isVector4 ) {
      return Matrix.rowVector4( vector );
    }
    else {
      throw new Error( "undetected type of vector: " + vector.toString() );
    }
  };

  Matrix.columnVector2 = function( vector ) {
    return new Matrix( 2, 1, [vector.x, vector.y] );
  };

  Matrix.columnVector3 = function( vector ) {
    return new Matrix( 3, 1, [vector.x, vector.y, vector.z] );
  };

  Matrix.columnVector4 = function( vector ) {
    return new Matrix( 4, 1, [vector.x, vector.y, vector.z, vector.w] );
  };

  Matrix.columnVector = function( vector ) {
    if ( vector.isVector2 ) {
      return Matrix.columnVector2( vector );
    }
    else if ( vector.isVector3 ) {
      return Matrix.columnVector3( vector );
    }
    else if ( vector.isVector4 ) {
      return Matrix.columnVector4( vector );
    }
    else {
      throw new Error( "undetected type of vector: " + vector.toString() );
    }
  };

  /**
   * Create a Matrix where each column is a vector
   */

  Matrix.fromVectors2 = function( vectors ) {
    var dimension = 2;
    var n = vectors.length;
    var data = new Float32Array( dimension * n );

    for ( var i = 0; i < n; i++ ) {
      var vector = vectors[i];
      data[i] = vector.x;
      data[i + n] = vector.y;
    }

    return new Matrix( dimension, n, data, true );
  };

  Matrix.fromVectors3 = function( vectors ) {
    var dimension = 3;
    var n = vectors.length;
    var data = new Float32Array( dimension * n );

    for ( var i = 0; i < n; i++ ) {
      var vector = vectors[i];
      data[i] = vector.x;
      data[i + n] = vector.y;
      data[i + 2 * n] = vector.z;
    }

    return new Matrix( dimension, n, data, true );
  };

  Matrix.fromVectors4 = function( vectors ) {
    var dimension = 4;
    var n = vectors.length;
    var data = new Float32Array( dimension * n );

    for ( var i = 0; i < n; i++ ) {
      var vector = vectors[i];
      data[i] = vector.x;
      data[i + n] = vector.y;
      data[i + 2 * n] = vector.z;
      data[i + 3 * n] = vector.w;
    }

    return new Matrix( dimension, n, data, true );
  };
  
  return Matrix;
} );

// Copyright 2002-2012, University of Colorado

/**
 * An immutable permutation that can permute an array
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Permutation',['require','ASSERT/assert','DOT/dot','PHET_CORE/isArray','DOT/Util'], function( require ) {
  
  
  var assert = require( 'ASSERT/assert' )( 'dot' );
  
  var dot = require( 'DOT/dot' );
  
  var isArray = require( 'PHET_CORE/isArray' );
  require( 'DOT/Util' ); // for rangeInclusive
  
  // Creates a permutation that will rearrange a list so that newList[i] = oldList[permutation[i]]
  var Permutation = function( indices ) {
    this.indices = indices;
  };

  // An identity permutation with a specific number of elements
  Permutation.identity = function( size ) {
    assert && assert( size >= 0 );
    var indices = new Array( size );
    for ( var i = 0; i < size; i++ ) {
      indices[i] = i;
    }
    return new Permutation( indices );
  };

  // lists all permutations that have a given size
  Permutation.permutations = function( size ) {
    var result = [];
    Permutation.forEachPermutation( dot.rangeInclusive( 0, size - 1 ), function( integers ) {
      result.push( new Permutation( integers ) );
    } );
    return result;
  };

  /**
   * Call our function with each permutation of the provided list PREFIXED by prefix, in lexicographic order
   *
   * @param array   List to generate permutations of
   * @param prefix   Elements that should be inserted at the front of each list before each call
   * @param callback Function to call
   */
  function recursiveForEachPermutation( array, prefix, callback ) {
    if ( array.length === 0 ) {
      callback.call( undefined, prefix );
    }
    else {
      for ( var i = 0; i < array.length; i++ ) {
        var element = array[i];

        // remove the element from the array
        var nextArray = array.slice( 0 );
        nextArray.splice( i, 1 );

        // add it into the prefix
        var nextPrefix = prefix.slice( 0 );
        nextPrefix.push( element );

        recursiveForEachPermutation( nextArray, nextPrefix, callback );
      }
    }
  }

  Permutation.forEachPermutation = function( array, callback ) {
    recursiveForEachPermutation( array, [], callback );
  };

  Permutation.prototype = {
    constructor: Permutation,

    size: function() {
      return this.indices.length;
    },

    apply: function( arrayOrInt ) {
      if ( isArray( arrayOrInt ) ) {
        if ( arrayOrInt.length !== this.size() ) {
          throw new Error( "Permutation length " + this.size() + " not equal to list length " + arrayOrInt.length );
        }

        // permute it as an array
        var result = new Array( arrayOrInt.length );
        for ( var i = 0; i < arrayOrInt.length; i++ ) {
          result[i] = arrayOrInt[ this.indices[i] ];
        }
        return result;
      }
      else {
        // permute a single index
        return this.indices[ arrayOrInt ];
      }
    },

    // The inverse of this permutation
    inverted: function() {
      var newPermutation = new Array( this.size() );
      for ( var i = 0; i < this.size(); i++ ) {
        newPermutation[this.indices[i]] = i;
      }
      return new Permutation( newPermutation );
    },

    withIndicesPermuted: function( indices ) {
      var result = [];
      var that = this;
      Permutation.forEachPermutation( indices, function( integers ) {
        var oldIndices = that.indices;
        var newPermutation = oldIndices.slice( 0 );

        for ( var i = 0; i < indices.length; i++ ) {
          newPermutation[indices[i]] = oldIndices[integers[i]];
        }
        result.push( new Permutation( newPermutation ) );
      } );
      return result;
    },

    toString: function() {
      return "P[" + this.indices.join( ", " ) + "]";
    }
  };

  Permutation.testMe = function( console ) {
    var a = new Permutation( [ 1, 4, 3, 2, 0 ] );
    console.log( a.toString() );

    var b = a.inverted();
    console.log( b.toString() );

    console.log( b.withIndicesPermuted( [ 0, 3, 4 ] ).toString() );

    console.log( Permutation.permutations( 4 ).toString() );
  };
  
  return Permutation;
} );

// Copyright 2002-2012, University of Colorado

/**
 * 3-dimensional ray
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Ray3',['require','DOT/dot'], function( require ) {
  
  
  var dot = require( 'DOT/dot' );
  
  dot.Ray3 = function( pos, dir ) {
    this.pos = pos;
    this.dir = dir;
  };
  var Ray3 = dot.Ray3;
  
  Ray3.prototype = {
    constructor: Ray3,

    shifted: function( distance ) {
      return new Ray3( this.pointAtDistance( distance ), this.dir );
    },

    pointAtDistance: function( distance ) {
      return this.pos.plus( this.dir.timesScalar( distance ) );
    },

    toString: function() {
      return this.pos.toString() + " => " + this.dir.toString();
    }
  };
  
  return Ray3;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Forward and inverse transforms with 4x4 matrices, allowing flexibility including affine and perspective transformations.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('DOT/Transform4',['require','DOT/dot','DOT/Matrix4','DOT/Vector3','DOT/Ray3'], function( require ) {
  
  
  var dot = require( 'DOT/dot' );
  
  require( 'DOT/Matrix4' );
  require( 'DOT/Vector3' );
  require( 'DOT/Ray3' );
  
  // takes a 4x4 matrix
  dot.Transform4 = function( matrix ) {
    // using immutable version for now. change it to the mutable identity copy if we need mutable operations on the matrices
    this.set( matrix === undefined ? dot.Matrix4.IDENTITY : matrix );
  };
  var Transform4 = dot.Transform4;
  
  Transform4.prototype = {
    constructor: Transform4,
    
    set: function( matrix ) {
      this.matrix = matrix;
      
      // compute these lazily
      this.inverse = null;
      this.matrixTransposed = null; // since WebGL won't allow transpose == true
      this.inverseTransposed = null;
    },
    
    getMatrix: function() {
      return this.matrix;
    },
    
    getInverse: function() {
      if ( this.inverse === null ) {
        this.inverse = this.matrix.inverted();
      }
      return this.inverse;
    },
    
    getMatrixTransposed: function() {
      if ( this.matrixTransposed === null ) {
        this.matrixTransposed = this.matrix.transposed();
      }
      return this.matrixTransposed;
    },
    
    getInverseTransposed: function() {
      if ( this.inverseTransposed === null ) {
        this.inverseTransposed = this.getInverse().transposed();
      }
      return this.inverseTransposed;
    },
    
    prepend: function( matrix ) {
      this.set( matrix.timesMatrix( this.matrix ) );
    },
    
    append: function( matrix ) {
      this.set( this.matrix.timesMatrix( matrix ) );
    },
    
    prependTransform: function( transform ) {
      this.prepend( transform.matrix );
    },
    
    appendTransform: function( transform ) {
      this.append( transform.matrix );
    },
    
    isIdentity: function() {
      return this.matrix.type === dot.Matrix4.Types.IDENTITY;
    },
    
    // applies the 2D affine transform part of the transformation
    applyToCanvasContext: function( context ) {
      context.setTransform( this.matrix.m00(), this.matrix.m10(), this.matrix.m01(), this.matrix.m11(), this.matrix.m03(), this.matrix.m13() );
    },
    
    /*---------------------------------------------------------------------------*
     * forward transforms (for Vector3 or scalar)
     *----------------------------------------------------------------------------*/
     
    // transform a position (includes translation)
    transformPosition3: function( vec3 ) {
      return this.matrix.timesVector3( vec3 );
    },
    
    // transform a vector (exclude translation)
    transformDelta3: function( vec3 ) {
      return this.matrix.timesRelativeVector3( vec3 );
    },
    
    // transform a normal vector (different than a normal vector)
    transformNormal3: function( vec3 ) {
      return this.getInverse().timesTransposeVector3( vec3 );
    },
    
    transformDeltaX: function( x ) {
      return this.transformDelta3( new dot.Vector3( x, 0, 0 ) ).x;
    },
    
    transformDeltaY: function( y ) {
      return this.transformDelta3( new dot.Vector3( 0, y, 0 ) ).y;
    },
    
    transformDeltaZ: function( z ) {
      return this.transformDelta3( new dot.Vector3( 0, 0, z ) ).z;
    },
    
    transformRay: function( ray ) {
      return new dot.Ray3(
          this.transformPosition3( ray.pos ),
          this.transformPosition3( ray.pos.plus( ray.dir ) ).minus( this.transformPosition3( ray.pos ) ) );
    },
    
    /*---------------------------------------------------------------------------*
     * inverse transforms (for Vector3 or scalar)
     *----------------------------------------------------------------------------*/
     
    inversePosition3: function( vec3 ) {
      return this.getInverse().timesVector3( vec3 );
    },
    
    inverseDelta3: function( vec3 ) {
      // inverse actually has the translation rolled into the other coefficients, so we have to make this longer
      return this.inversePosition3( vec3 ).minus( this.inversePosition3( dot.Vector3.ZERO ) );
    },
    
    inverseNormal3: function( vec3 ) {
      return this.matrix.timesTransposeVector3( vec3 );
    },
    
    inverseDeltaX: function( x ) {
      return this.inverseDelta3( new dot.Vector3( x, 0, 0 ) ).x;
    },
    
    inverseDeltaY: function( y ) {
      return this.inverseDelta3( new dot.Vector3( 0, y, 0 ) ).y;
    },
    
    inverseDeltaZ: function( z ) {
      return this.inverseDelta3( new dot.Vector3( 0, 0, z ) ).z;
    },
    
    inverseRay: function( ray ) {
      return new dot.Ray3(
          this.inversePosition3( ray.pos ),
          this.inversePosition3( ray.pos.plus( ray.dir ) ).minus( this.inversePosition3( ray.pos ) )
      );
    }
  };
  
  return Transform4;
} );


define('DOT/main', [
  'DOT/dot',
  'DOT/Bounds2',
  'DOT/ConvexHull2',
  'DOT/Dimension2',
  'DOT/LUDecomposition',
  'DOT/Matrix',
  'DOT/Matrix3',
  'DOT/Matrix4',
  'DOT/Permutation',
  'DOT/QRDecomposition',
  'DOT/Ray2',
  'DOT/Ray3',
  'DOT/SingularValueDecomposition',
  'DOT/Transform3',
  'DOT/Transform4',
  'DOT/Util',
  'DOT/Vector2',
  'DOT/Vector3',
  'DOT/Vector4'
  ], function( dot ) {
  return dot;
} );

// Copyright 2013, University of Colorado

/**
 * A method of calling an overridden super-type method.
 *
 * @author Chris Malley (PixelZoom, Inc.)
 */
define('PHET_CORE/callSuper',['require'], function( require ) {
  

  /**
   * A somewhat ugly method of calling an overridden super-type method.
   * <p>
   * Example:
   * <code>
   * function SuperType() {
   * }
   *
   * SuperType.prototype.reset = function() {...}
   *
   * function SubType() {
   *    SuperType.call( this ); // constructor stealing
   * }
   *
   * SubType.prototype = new SuperType(); // prototype chaining
   *
   * SubType.prototype.reset = function() {
   *     Inheritance.callSuper( SuperType, "reset", this ); // call overridden super method
   *     // do subtype-specific stuff
   * }
   * </code>
   *
   * @param supertype
   * @param {String} name
   * @param context typically this
   * @return {Function}
   */
  function callSuper( supertype, name, context ) {
    (function () {
      var fn = supertype.prototype[name];
      Function.call.apply( fn, arguments );
    })( context );
  }

  return callSuper;
} );

// Copyright 2013, University of Colorado

/**
 * Prototype chaining using Parasitic Combination Inheritance
 *
 * @author Chris Malley (PixelZoom, Inc.)
 */
define('PHET_CORE/inheritPrototype',['require'], function( require ) {
  

  /**
   * Use this function to do prototype chaining using Parasitic Combination Inheritance.
   * Instead of calling the supertype's constructor to assign a prototype (as is done
   * in Combination Inheritance), you create a copy of the supertype's prototype.
   * <br>
   * Here's the basic pattern:
   * <br>
   * <code>
   * function Supertype(...) {...}
   *
   * function Subtype(...) {
           *     Supertype.call(this, ...); // constructor stealing, called second
           *     ...
           * }
   *
   * inheritPrototype( Subtype, Supertype ); // prototype chaining, called first
   * </code>
   * <br>
   * (source: JavaScript for Web Developers, N. Zakas, Wrox Press, p. 212-215)
   */
  function inheritPrototype( subtype, supertype ) {
    var prototype = Object( supertype.prototype ); // create a clone of the supertype's prototype
    prototype.constructor = subtype; // account for losing the default constructor when prototype is overwritten
    subtype.prototype = prototype; // assign cloned prototype to subtype
  }

  return inheritPrototype;
} );

// Copyright 2002-2012, University of Colorado

/**
 * Loads a script
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define('PHET_CORE/loadScript',['require'], function( require ) {
  
  
  /*
   * Load a script. The only required argument is src, and can be specified either as
   * loadScript( "<url>" ) or loadScript( { src: "<url>", ... other options ... } ).
   *
   * Arguments:
   *   src:         The source of the script to load
   *   callback:    A callback to call (with no arguments) once the script is loaded and has been executed
   *   async:       Whether the script should be loaded asynchronously. Defaults to true
   *   cacheBuster: Whether the URL should have an appended query string to work around caches
   */
  return function loadScript( args ) {
    // handle a string argument
    if ( typeof args === 'string' ) {
      args = { src: args };
    }
    
    var src         = args.src;
    var callback    = args.callback;
    var async       = args.async === undefined ? true : args.async;
    var cacheBuster = args.cacheBuster === undefined ? false : args.cacheBuster;
    
    var called = false;
    
    var script = document.createElement( 'script' );
    script.type = 'text/javascript';
    script.async = async;
    script.onload = script.onreadystatechange = function() {
      var state = this.readyState;
      if ( state && state !== "complete" && state !== "loaded" ) {
        return;
      }
      
      if ( !called ) {
        called = true;
        
        if ( callback ) {
          callback();
        }
      }
    };
    
    // make sure things aren't cached, just in case
    script.src = src + ( cacheBuster ? '?random=' + Math.random().toFixed( 10 ) : '' );
    
    var other = document.getElementsByTagName( 'script' )[0];
    other.parentNode.insertBefore( script, other );
  };
} );


define('PHET_CORE/main',['require','PHET_CORE/callSuper','PHET_CORE/inherit','PHET_CORE/inheritPrototype','PHET_CORE/isArray','PHET_CORE/extend','PHET_CORE/loadScript'], function( require ) {
  return {
    callSuper: require( 'PHET_CORE/callSuper' ),
    inherit: require( 'PHET_CORE/inherit' ),
    inheritPrototype: require( 'PHET_CORE/inheritPrototype' ),
    isArray: require( 'PHET_CORE/isArray' ),
    extend: require( 'PHET_CORE/extend' ),
    loadScript: require( 'PHET_CORE/loadScript' )
  };
} );

// Copyright 2002-2012, University of Colorado

/**
 * Configuration file for development purposes, NOT for production deployments.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

// if has.js is included, set assertion flags to true (so we can catch errors during development)
if ( window.has ) {
  window.has.add( 'assert.dot', function( global, document, anElement ) {
    return true;
  } );
  window.has.add( 'assert.kite', function( global, document, anElement ) {
    return true;
  } );
  window.has.add( 'assert.kite.extra', function( global, document, anElement ) {
    return true;
  } );
  window.has.add( 'assert.scenery', function( global, document, anElement ) {
    return true;
  } );
  window.has.add( 'assert.scenery.extra', function( global, document, anElement ) {
    return true;
  } );
}

// flag is set so we can ensure that the config has executed. This prevents various Require.js dynamic loading timeouts and script errors
window.loadedSceneryConfig = true;

require.config( {
  // depends on all of Scenery, Kite, and Dot
  deps: [ 'main', 'KITE/main', 'DOT/main', 'PHET_CORE/main' ],
  
  paths: {
    underscore: '../contrib/lodash.min-1.0.0-rc.3',
    jquery: '../contrib/jquery-1.8.3.min',
    SCENERY: '.',
    KITE: '../common/kite/js',
    DOT: '../common/dot/js',
    PHET_CORE: '../common/phet-core/js',
    ASSERT: '../common/assert/js'
  },
  
  shim: {
    underscore: { exports: '_' },
    jquery: { exports: '$' }
  },
  
  urlArgs: new Date().getTime() // add cache buster query string to make browser refresh actually reload everything
} );

define("config", function(){});
 window.scenery = require( 'main' ); window.kite = require( 'KITE/main' ); window.dot = require( 'DOT/main' ); window.core = require( 'PHET_CORE/main' ); scenery.Util.polyfillRequestAnimationFrame(); }());
