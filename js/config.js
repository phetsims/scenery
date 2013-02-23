
require.config( {
  deps: [ 'main', 'DOT/main' ],
  
  paths: {
    underscore: '../contrib/lodash.min-1.0.0-rc.3',
    jquery: '../contrib/jquery-1.8.3.min',
    SCENERY: '.',
    DOT: '../common/dot/js',
    ASSERT: '../common/assert/js'
  },
  
  shim: {
    underscore: {
      exports: '_'
    },
    jquery: {
      exports: '$'
    }
  },
  
  urlArgs: new Date().getTime() // add cache buster query string to make browser refresh actually reload everything
} );
