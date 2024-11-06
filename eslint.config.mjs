// Copyright 2024, University of Colorado Boulder

/**
 * ESLint configuration for scenery.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import phetLibraryEslintConfig from '../perennial-alias/js/eslint/config/phet-library.eslint.config.mjs';

export default [
  ...phetLibraryEslintConfig,
  {
    rules: {
      'no-bitwise': 'off'
    },
    languageOptions: {
      globals: {
        himalaya: 'readonly',
        LineBreaker: 'readonly',
        sceneryLog: 'readonly',
        he: 'readonly'
      }
    }
  },
  {
    files: [
      'tests/**/*.html',
      'examples/**/*.html'
    ],
    rules: {
      'phet/bad-sim-text': 'off'
    }
  },
  {
    ignores: [
      'js/display/swash/pkg',
      'js/display/guillotiere/pkg'
    ]
  }
];