// Copyright 2002-2015, University of Colorado Boulder

var exec = require( 'child_process' ).exec;

/*global module:false*/
module.exports = function( grunt ) {
  'use strict';

  // print this immediately, so it is clear what project grunt is building
  grunt.log.writeln( 'Scenery' );

  // --disable-es-cache disables the cache, useful for developing rules
  var cache = !grunt.option( 'disable-eslint-cache' );

  // Project configuration.
  grunt.initConfig( {
    pkg: '<json:package.json>',

    requirejs: {
      // unminified
      development: {
        options: {
          almond: true,
          mainConfigFile: 'js/config.js',
          out: 'build/development/scenery.js',
          name: 'config',
          optimize: 'none',
          wrap: {
            startFile: [ 'js/wrap-start.frag', '../assert/js/assert.js' ],
            endFile: [ 'js/wrap-end.frag' ]
          }
        }
      },

      production: {
        options: {
          almond: true,
          mainConfigFile: 'js/config.js',
          out: 'build/production/scenery.min.js',
          name: 'config',
          optimize: 'uglify2',
          generateSourceMaps: true,
          preserveLicenseComments: false,
          wrap: {
            startFile: [ 'js/wrap-start.frag', '../assert/js/assert.js' ],
            endFile: [ 'js/wrap-end.frag' ]
          },
          uglify2: {
            compress: {
              global_defs: {
                assert: false,
                assertSlow: false,
                sceneryLog: false,
                sceneryAccessibilityLog: false,
                phetAllocation: false
              },
              dead_code: true
            }
          }
        }
      }
    },

    eslint: {
      options: {

        // Rules are specified in the .eslintrc file
        configFile: '../chipper/eslint/.eslintrc',

        // Caching only checks changed files or when the list of rules is changed.  Changing the implementation of a
        // custom rule does not invalidate the cache.  Caches are declared in .eslintcache files in the directory where
        // grunt was run from.
        cache: cache,

        // Our custom rules live here
        rulePaths: [ '../chipper/eslint/rules' ]
      },

      files: [
        'Gruntfile.js',
        '../phet-core/js/**/*.js',
        '../axon/js/**/*.js',
        '../dot/js/**/*.js',
        '../kite/js/**/*.js',
        '../assert/js/**/*.js',
        'js/**/*.js',
        '!../kite/js/parser/svgPath.js'
      ]
    }
  } );

  // default task ('grunt')
  grunt.registerTask( 'default', [ 'lint', 'development', 'production' ] );

  grunt.registerTask( 'lint', [ 'eslint:files' ] );

  // compilation targets. invoke only one like ('grunt development')
  grunt.registerTask( 'production', [ 'requirejs:production' ] );
  grunt.registerTask( 'development', [ 'requirejs:development' ] );

  grunt.registerTask( 'snapshot', [ 'production', '_createSnapshot' ] );

  // creates a performance snapshot for profiling changes
  grunt.registerTask( '_createSnapshot', 'Description', function( arg ) {
    var done = this.async();

    exec( 'git log -1 --date=short', function( error, stdout, stderr ) {
      if ( error ) { throw error; }

      var sha = /commit (.*)$/m.exec( stdout )[ 1 ];
      var date = /Date: *(\d+)-(\d+)-(\d+)$/m.exec( stdout );
      var year = date[ 1 ].slice( 2, 4 );
      var month = date[ 2 ];
      var day = date[ 3 ];

      var suffix = '-' + year + month + day + '-' + sha.slice( 0, 10 ) + '.js';

      grunt.file.copy( 'build/production/scenery.min.js', 'snapshots/scenery-min' + suffix );
      grunt.file.copy( 'tests/benchmarks/js/perf-current.js', 'snapshots/perf' + suffix );

      grunt.log.writeln( 'Copied production js to snapshots/scenery-min' + suffix );
      grunt.log.writeln( 'Copied perf js to       snapshots/perf' + suffix );

      done();
    } );
  } );

  // dependencies
  grunt.loadNpmTasks( 'grunt-requirejs' );
  grunt.loadNpmTasks( 'grunt-eslint' );
};
