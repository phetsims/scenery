// Copyright 2024, University of Colorado Boulder

/**
 * ESlint configuration for scenery.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import phetLibraryEslintConfig from '../chipper/eslint/phet-library.eslint.config.mjs';

export default [
  ...phetLibraryEslintConfig,
  {
    rules: {
      'no-bitwise': 'off'
    },
    ignores: [
      'js/display/swash/pkg',
      'js/display/guillotiere/pkg'
    ],
    languageOptions: {
      globals: {
        himalaya: 'readonly',
        LineBreaker: 'readonly',
        sceneryLog: 'readonly',
        he: 'readonly'
      }
    }
  }
];