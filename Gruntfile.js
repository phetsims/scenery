/*global module:false*/
module.exports = function( grunt ) {
  
  // Project configuration.
  grunt.initConfig( {
    pkg: '<json:package.json>',
    
    lint: {
      files: [
       // 'grunt.js',
       // 'app-easel/*.js'
      ]
    },
    
    concat: {
      standalone: {
        src: [ "contrib/almond.js", "dist/standalone/scenery.js" ],
        dest: "dist/standalone/scenery.js"
      }
    },
    
    uglify: {
      standalone: {
        src: [ 'dist/standalone/scenery.js' ],
        dest: 'dist/standalone/scenery.min.js'
      }
    },
    
    requirejs: {
      standalone: {
        options: {
          mainConfigFile: "js/config.js",
          out: "dist/standalone/scenery.js",
          name: "config",
          // optimize: 'uglify2',
          optimize: 'none',
          wrap: {
            start: "(function() {",
            end: " window.scenery = require( 'main' ); window.dot = require( 'DOT/main' ); }());"
          }
        }
      },
      production: {
        options: {
          mainConfigFile: "js/config.js",
          out: "dist/production/scenery.min.js",
          name: "config",
          optimize: 'uglify2',
          generateSourceMaps: true,
          preserveLicenseComments: false,
          wrap: {
            start: "(function() {",
            end: " window.scenery = require( 'main' ); }());"
          }
        }
      }
    },
    
    jshint: {
      all: [
        'Gruntfile.js', 'js/**/*.js'
      ],
      // adjust with options from http://www.jshint.com/docs/
      options: {
        // enforcing options
        curly: true, // brackets for conditionals
        eqeqeq: true,
        immed: true,
        latedef: true,
        newcap: true,
        noarg: true,
        // noempty: true,
        nonew: true,
        // quotmark: 'single',
        undef: true,
        // unused: true, // certain layer APIs not used in cases
        // strict: true,
        trailing: true,
        
        // relaxing options
        es5: true, // we use ES5 getters and setters for now
        loopfunc: true, // we know how not to shoot ourselves in the foot, and this is useful for _.each
        
        expr: true, // so we can use assert && assert( ... )
        
        globals: {
          // for require.js
          define: true,
          require: true,
          
          Uint16Array: false,
          Uint32Array: false,
          document: false,
          window: false,
          console: false,
          Float32Array: true, // we actually polyfill this, so allow it to be set
          
          $: false,
          _: false,
          clearTimeout: false,
          
          // for DOM.js
          Image: false,
          Blob: false,
          
          canvg: false
        }
      },
    }
  } );
  
  // Default task.
  grunt.registerTask( 'default', [ 'jshint', 'production', 'standalone' ] );
  grunt.registerTask( 'production', [ 'requirejs:production' ] );
  grunt.registerTask( 'standalone', [ 'requirejs:standalone', 'concat:standalone', 'uglify:standalone' ] );
  grunt.loadNpmTasks( 'grunt-contrib-requirejs' );
  grunt.loadNpmTasks( 'grunt-contrib-concat' );
  grunt.loadNpmTasks( 'grunt-contrib-uglify' );
  grunt.loadNpmTasks( 'grunt-contrib-jshint' );
};
